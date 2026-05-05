import argparse
import json
import os
import csv
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import NAdam
from tqdm import tqdm
from dotenv import load_dotenv

from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning import losses
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Custom imports
from lvm_utils.utils import materialize_conversation_images
from lvm_utils.model_helpers import load_model_id, build_conversation_embedding_config, get_batch_conversation_embeddings_with_config
from lvm_utils.mc_head import MonteCarloDropoutHead

class VLMDataset(Dataset):
    def __init__(self, index_df, cache_dir='./cache_data', label_mapping=None):
        self.df = index_df[index_df['status'].isin(['done', 'STATUS_DONE'])].reset_index(drop=True)
        self.cache_dir = cache_dir
        
        if 'label' not in self.df.columns:
            raise KeyError("The index dataframe must contain a 'label' column.")
            
        if label_mapping is None:
            self.labels, self.label_uniques = pd.factorize(self.df['label'])
            self.label_mapping = {val: i for i, val in enumerate(self.label_uniques)}
        else:
            self.label_mapping = label_mapping
            mapped_labels = self.df['label'].map(self.label_mapping)
            unknown_mask = mapped_labels.isna()
            if unknown_mask.any():
                self.df = self.df.loc[~unknown_mask].reset_index(drop=True)
                mapped_labels = mapped_labels.loc[~unknown_mask].reset_index(drop=True)
            self.labels = mapped_labels.astype('int64').to_numpy()
            self.label_uniques = pd.Series(list(self.label_mapping.keys()))
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        conv_path = os.path.join(self.cache_dir, row['conversation_json_path'])
        with open(conv_path, "r") as f:
            raw_conv = json.load(f)
        
        conv = materialize_conversation_images(raw_conv, self.cache_dir)
        return conv, self.labels[idx]

def collate_fn(batch):
    convs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return convs, labels

def main():
    parser = argparse.ArgumentParser(description="Run LVM Training with PEFT using a config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        config = json.load(f)

    load_dotenv()
    
    experiment_name = config.get("experiment_name", "experiment")
    run_dir = os.path.join("runs", experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save a copy of the config in the run directory for reproducibility
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
        
    metrics_csv_path = os.path.join(run_dir, "metrics.csv")
    with open(metrics_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_acc", "head_acc"])

    print(f"Starting experiment: {experiment_name}")

    # Load Model
    model, processor = load_model_id(load_peft=True)
    model.gradient_checkpointing_enable()

    # Load Data
    index_path = config.get("index_path", "cache_data/index.parquet")
    index = pd.read_parquet(index_path)

    train_index, val_index = train_test_split(
        index,
        test_size=config.get("test_size", 0.1),
        random_state=42,
        stratify=index['label'],
    )

    cache_dir = config.get("cache_dir", "./cache_data")
    train_dataset = VLMDataset(train_index, cache_dir=cache_dir)
    val_dataset = VLMDataset(val_index, cache_dir=cache_dir, label_mapping=train_dataset.label_mapping)

    batch_size = config.get("batch_size", 5)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    try:
        embedding_size = model.config.text_config.hidden_size
    except AttributeError:
        embedding_size = 1024

    output_embedding_size = config.get("output_embedding_size", 368)
    num_classes = len(train_dataset.label_uniques)

    print(f"Initializing ProxyAnchorLoss: classes={num_classes}, embed_dim={embedding_size}")

    head = MonteCarloDropoutHead(
        embedding_size, output_embedding_size, num_classes,
        dropout_prob=config.get("head_dropout_prob", 0.3)
    )
    head.to(model.device)

    training_mode = config.get("training_mode", "proxy_anchor").lower()

    loss_func = None
    if training_mode in ("proxy_anchor", "joint"):
        loss_func = losses.ProxyAnchorLoss(
            num_classes=num_classes, 
            embedding_size=output_embedding_size, 
            margin=0.1, 
            alpha=32
        ).to(model.device)

    trainable_model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer_param_groups = [
        {"params": trainable_model_params, "lr": config.get("learning_rate_model", 1e-5)},
        {"params": head.parameters(), "lr": config.get("learning_rate_head", 1e-3)}
    ]
    if loss_func is not None:
        optimizer_param_groups.append(
            {"params": loss_func.parameters(), "lr": config.get("learning_rate_head", 1e-3)}
        )
    optimizer = NAdam(optimizer_param_groups)

    epochs = config.get("epochs", 20)
    accumulation_steps = config.get("accumulation_steps", 8)
    intermediate_reasoning_drop_probability = config.get("intermediate_reasoning_drop_probability", 0.3)
    checkpoint_interval = config.get("checkpoint_interval", 5)
    eval_interval = config.get("eval_interval", 1)
    ce_loss_weight = config.get("ce_loss_weight", 0.3)

    llm_eos_config = build_conversation_embedding_config(processor)

    tb_writer = SummaryWriter(log_dir=run_dir)
    global_step = 0

    print("Beginning Training Loop...")
    for epoch in range(epochs):
        model.train()
        head.train()
        torch.cuda.empty_cache()

        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        epoch_loss = 0.0
        
        progress = tqdm(trainloader, desc="Training")
        optimizer.zero_grad(set_to_none=True)
        
        for i, (batch_convs, batch_labels) in enumerate(progress):
            batch_labels = batch_labels.to(model.device)
            
            with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16):
                vecs_o, lens = get_batch_conversation_embeddings_with_config(
                    model=model, 
                    processor=processor, 
                    conversations=batch_convs, 
                    config=llm_eos_config,
                    normalize=True,
                    intermediate_reasoning_drop_probability=intermediate_reasoning_drop_probability,
                )
                preds, vecs = head(vecs_o)
                
                if training_mode == "joint":
                    ce_loss = torch.nn.functional.cross_entropy(preds, batch_labels)
                    loss = loss_func(vecs, batch_labels) + ce_loss_weight * ce_loss
                elif training_mode == "classification":
                    loss = torch.nn.functional.cross_entropy(preds, batch_labels)
                else:  # proxy_anchor
                    loss = loss_func(vecs, batch_labels)
                
            loss = loss / accumulation_steps
            loss.backward()
            
            if ((i + 1) % accumulation_steps == 0) or (i + 1 == len(trainloader)):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            real_loss = loss.item() * accumulation_steps
            epoch_loss += real_loss
            progress.set_postfix({"loss": f"{real_loss:.4f}"})
            
            tb_writer.add_scalar('Loss/train_step', real_loss, global_step)
            global_step += 1
            
            del vecs, loss, batch_convs, batch_labels
        
        avg_train_loss = epoch_loss / len(trainloader)
        tb_writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        print(f"Epoch {epoch+1} Complete | Average Train Loss: {avg_train_loss:.4f}")

        val_acc = 0.0
        head_acc = 0.0
        # Evaluation
        if (epoch + 1) % eval_interval == 0:
            model.eval()
            head.eval()
            
            train_embeddings = []
            train_targets = []
            val_embeddings = []
            val_targets = []
            
            with torch.no_grad():
                train_progress = tqdm(trainloader, desc="Extracting Train Embeddings")
                for batch_convs, batch_labels in train_progress:
                    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16):
                        vecs_o, lens = get_batch_conversation_embeddings_with_config(
                            model=model, processor=processor, conversations=batch_convs, 
                            config=llm_eos_config, normalize=True
                        )
                        _, vecs = head(vecs_o)
                    train_embeddings.append(vecs.float().cpu())
                    train_targets.append(batch_labels.cpu())
                    del vecs, batch_convs, batch_labels, vecs_o
                    
                train_embeddings_tensor = torch.cat(train_embeddings)
                train_targets_tensor = torch.cat(train_targets)
                
                val_preds_list = []
                val_progress = tqdm(valloader, desc="Extracting Val Embeddings")
                for batch_convs, batch_labels in val_progress:
                    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16):
                        vecs_o, lens = get_batch_conversation_embeddings_with_config(
                            model=model, processor=processor, conversations=batch_convs, 
                            config=llm_eos_config, normalize=True
                        )
                        preds, vecs = head(vecs_o)
                    val_embeddings.append(vecs.float().cpu())
                    val_targets.append(batch_labels.cpu())
                    val_preds_list.append(preds.float().cpu())
                    del preds, vecs, batch_convs, batch_labels, vecs_o
                    
                val_embeddings_tensor = torch.cat(val_embeddings)
                val_targets_tensor = torch.cat(val_targets)
                val_preds_tensor = torch.cat(val_preds_list)
                
                # Head classification accuracy (always computed)
                predicted_labels = val_preds_tensor.argmax(dim=1)
                head_acc = (predicted_labels == val_targets_tensor).float().mean().item()
                tb_writer.add_scalar('Acc/val_head', head_acc, epoch)
                print(f"Epoch {epoch+1} Complete | Head Classification Val Accuracy: {head_acc:.4f}")
                
            evaluation_method = config.get("evaluation_method", "knn").lower()
            
            if evaluation_method == "linear_probe":
                clf = LogisticRegression(
                    max_iter=config.get("linear_probe_max_iter", 1000), 
                    n_jobs=-1
                )
                clf.fit(train_embeddings_tensor.numpy(), train_targets_tensor.numpy())
                val_acc = clf.score(val_embeddings_tensor.numpy(), val_targets_tensor.numpy())
                
                tb_writer.add_scalar('Acc/val_linear_probe', val_acc, epoch)
                print(f"Epoch {epoch+1} Complete | Linear Probe Val Accuracy: {val_acc:.4f}")
                
            else: # Fallback to k-NN
                knn = KNeighborsClassifier(n_neighbors=config.get("k_neighbors", 5))
                knn.fit(train_embeddings_tensor.numpy(), train_targets_tensor.numpy())
                val_acc = knn.score(val_embeddings_tensor.numpy(), val_targets_tensor.numpy())
                
                tb_writer.add_scalar('Acc/val_knn', val_acc, epoch)
                print(f"Epoch {epoch+1} Complete | k-NN Val Accuracy: {val_acc:.4f}")

        # Logging Metrics to CSV
        with open(metrics_csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, avg_train_loss, val_acc, head_acc])

        # Checkpointing
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == epochs:
            ckpt_dir = os.path.join(run_dir, f"checkpoint-{epoch+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            print(f"Saving checkpoint to {ckpt_dir}...")
            
            # Save LoRA adapter
            model.save_pretrained(ckpt_dir)
            
            # Save MC Head, ProxyAnchorLoss params if needed
            torch.save(head.state_dict(), os.path.join(ckpt_dir, "mc_head.pth"))
            if loss_func is not None:
                torch.save(loss_func.state_dict(), os.path.join(ckpt_dir, "proxy_anchor_loss.pth"))
            
            print(f"Checkpoint saved successfully.")

    tb_writer.close()
    print("Training completely finished!")

if __name__ == "__main__":
    main()
