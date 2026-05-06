#%%
import argparse
import json
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, NAdam
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
from lvm_utils.utils import materialize_conversation_images
from lvm_utils.model_helpers import load_model_id, build_conversation_embedding_config, get_batch_conversation_embeddings_with_config

from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from pytorch_metric_learning import losses
from sklearn.neighbors import KNeighborsClassifier
import gc

#%%
# Parse arguments (use parse_known_args so the file can still be run from
# Jupyter / interactive environments where extra args may be injected)
parser = argparse.ArgumentParser(description="Train LVM with PEFT.")
parser.add_argument("--resume_from", type=str, default=None,
                    help="Path to a checkpoint directory to resume training from.")
parser.add_argument("--checkpoint_interval", type=int, default=5,
                    help="Save a checkpoint every N epochs.")
args, _ = parser.parse_known_args()

resume_from = args.resume_from
checkpoint_interval = args.checkpoint_interval

#%%
model, processor = load_model_id(load_peft=True)
# model.gradient_checkpointing_enable()

# %%
# Build a proper Dataset and DataLoader for the Training Loop
class VLMDataset(Dataset):
    def __init__(self, index_df, cache_dir='./cache_data', label_mapping=None):
        # Only keep successfully processed ones that have labels
        self.df = index_df[index_df['status'].isin(['done', 'STATUS_DONE'])].reset_index(drop=True)
        self.cache_dir = cache_dir

        if 'label' not in self.df.columns:
            raise KeyError("The index dataframe must contain a 'label' column.")

        # Convert string labels to 0-indexed integers for ProxyAnchorLoss
        if label_mapping is None:
            self.labels, self.label_uniques = pd.factorize(self.df['label'])
            self.label_mapping = {val: i for i, val in enumerate(self.label_uniques)}
        else:
            self.label_mapping = label_mapping
            self.labels = self.df['label'].map(self.label_mapping).values
            self.label_uniques = pd.Series(list(self.label_mapping.keys()))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        conv_path = self.cache_dir + "/" + row['conversation_json_path']
        with open(conv_path, "r") as f:
            raw_conv = json.load(f)

        # Load the PIL images back into the dict
        # Ensure your materialization reads dict refs cleanly
        conv = materialize_conversation_images(raw_conv, self.cache_dir)
        return conv, self.labels[idx]

def collate_fn(batch):
    convs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return convs, labels

# %%
# Load index and initialize Dataset
train_index = pd.read_parquet("cached_cifar100/index.parquet")
test_index = pd.read_parquet("cached_cifar100_test/index.parquet")
# Split index into train and validation (90% train, 10% val)
# val_index = index.sample(frac=0.1, random_state=42)
# train_index = index.drop(val_index.index)
train_index = train_index.sample(frac=0.3)
test_index = test_index.sample(frac=0.01)
train_dataset = VLMDataset(train_index, cache_dir = "cached_cifar100")
val_dataset = VLMDataset(test_index, label_mapping=train_dataset.label_mapping, cache_dir = "cached_cifar100_test")

batch_size = 20  # Keep this small to avoid VRAM OOM! (12GB -> ~4-8 depending on token length)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# %%
# Setup ProxyAnchorLoss & Optimizer
# LFM2-VL-450M text config has hidden_size=1024
from lvm_utils.mc_head import MonteCarloDropoutHead



try:
    input_embedding_size = model.config.text_config.hidden_size
except AttributeError:
    input_embedding_size = 1024 # Fallback

output_embedding_size = 368
num_classes = len(train_dataset.label_uniques)
head = MonteCarloDropoutHead(input_embedding_size, output_embedding_size,num_classes, dropout_prob=0.3)
head.to(model.device)

print(f"Initializing ProxyAnchorLoss: classes={num_classes}, embed_dim={output_embedding_size}")

loss_func = losses.ProxyAnchorLoss(
    num_classes=num_classes, 
    embedding_size=output_embedding_size, 
    margin=0.1, 
    alpha=32
).to(model.device)
# sup_con_loss = losses.SupConLoss(temperature=0.1)
# loss_func = losses.CrossBatchMemory(loss_func, embedding_size, memory_size=40, miner=None)

# Ensure we optimize both model (PEFT params) AND the ProxyAnchor embeddings
trainable_model_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = NAdam([
    {"params": trainable_model_params, "lr": 5e-4}, 
    {"params": loss_func.parameters(), "lr": 5e-3},
    {"params" : head.parameters(), "lr" : 5e-4}
],decoupled_weight_decay = True,
    weight_decay = 5e-4)
llm_eos_config = build_conversation_embedding_config(processor)

def _resolve_checkpoint_dir(path: str) -> str:
    """Resolve a path to the actual checkpoint directory containing adapter weights.

    Accepts either:
    - A direct checkpoint directory (``checkpoint-N/``) containing adapter weights.
    - A run directory whose subdirectories contain ``checkpoint-N/`` folders;
      in this case the subdirectory with the highest N is selected automatically.

    Raises FileNotFoundError if no valid checkpoint can be found.
    """
    if (os.path.isfile(os.path.join(path, "adapter_model.safetensors")) or
            os.path.isfile(os.path.join(path, "adapter_model.bin"))):
        return path

    candidates = []
    for entry in os.scandir(path):
        if entry.is_dir() and entry.name.startswith("checkpoint-"):
            try:
                epoch_num = int(entry.name.split("-")[-1])
                candidates.append((epoch_num, entry.path))
            except ValueError:
                pass

    if not candidates:
        raise FileNotFoundError(
            f"No adapter weights or checkpoint subdirectories found in '{path}'. "
            "Provide a path to a specific 'checkpoint-N' directory or a run directory "
            "that contains such subdirectories."
        )

    candidates.sort(key=lambda x: x[0])
    latest_epoch, latest_path = candidates[-1]
    print(f"  Auto-selected latest checkpoint: {latest_path} (epoch {latest_epoch})")
    return latest_path


def _load_peft_adapter(model, ckpt_dir: str):
    """Load saved PEFT adapter weights into an already-initialised PeftModel.

    Tries ``adapter_model.safetensors`` first, then ``adapter_model.bin``.
    Uses ``peft.set_peft_model_state_dict`` so the adapter name does not need
    to be known by the caller.
    """
    from peft import set_peft_model_state_dict

    safetensors_path = os.path.join(ckpt_dir, "adapter_model.safetensors")
    bin_path = os.path.join(ckpt_dir, "adapter_model.bin")

    if os.path.isfile(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
    elif os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No adapter weights found in '{ckpt_dir}'. "
            "Expected 'adapter_model.safetensors' or 'adapter_model.bin'."
        )

    set_peft_model_state_dict(model, state_dict)


#%%
# Resume from checkpoint (if requested)
run_dir = "./runs/lvm_training"
os.makedirs(run_dir, exist_ok=True)

start_epoch = 0
global_step = 0

if resume_from:
    resume_from = _resolve_checkpoint_dir(resume_from)
    print(f"Resuming training from checkpoint: {resume_from}")

    # Restore PEFT adapter weights
    _load_peft_adapter(model, resume_from)
    print("  Loaded PEFT adapter weights.")

    # Restore MC head
    head_ckpt = os.path.join(resume_from, "mc_head.pth")
    if os.path.isfile(head_ckpt):
        head.load_state_dict(torch.load(head_ckpt, map_location=model.device))
        print("  Loaded MC head weights.")

    # Restore ProxyAnchor loss proxies
    loss_ckpt = os.path.join(resume_from, "proxy_anchor_loss.pth")
    if os.path.isfile(loss_ckpt):
        loss_func.load_state_dict(torch.load(loss_ckpt, map_location=model.device))
        print("  Loaded ProxyAnchorLoss weights.")

    # Restore optimizer
    opt_ckpt = os.path.join(resume_from, "optimizer.pth")
    if os.path.isfile(opt_ckpt):
        optimizer.load_state_dict(torch.load(opt_ckpt, map_location="cpu"))
        print("  Loaded optimizer state.")

    # Restore training progress
    state_path = os.path.join(resume_from, "training_state.json")
    if os.path.isfile(state_path):
        with open(state_path, "r") as f:
            training_state = json.load(f)
        start_epoch = training_state.get("epoch", 0)
        global_step = training_state.get("global_step", 0)
        print(f"  Resuming from epoch {start_epoch}, global step {global_step}.")
    else:
        # Infer epoch from directory name (e.g. "checkpoint-10" -> start at epoch 10)
        dir_name = os.path.basename(resume_from.rstrip("/"))
        if dir_name.startswith("checkpoint-"):
            try:
                start_epoch = int(dir_name.split("-")[-1])
                print(f"  Inferred start epoch {start_epoch} from directory name (no training_state.json found).")
            except ValueError:
                pass

#%%
# Training Loop
epochs = 20
accumulation_steps = 5 # Simulate a larger batch size (e.g. batch_size 4 * acc_steps 4 = effective batch 16)
llm_eos_config = build_conversation_embedding_config(processor)

model.train()
torch.cuda.empty_cache()

writer = SummaryWriter(log_dir=run_dir)

for epoch in range(start_epoch, epochs):
    print(f"\n--- Epoch {epoch+1}/{epochs} ---")
    epoch_loss = 0.0

    progress = tqdm(trainloader, desc="Training")
    optimizer.zero_grad(set_to_none=True)

    for i, (batch_convs, batch_labels) in enumerate(progress):
        batch_labels = batch_labels.to(model.device)

        # Mixed Precision Context for fast 4-bit/bfloat16 evaluation
        with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16):
            # 1. Extract Contrastive Embeddings
            vecs_o, lens = get_batch_conversation_embeddings_with_config(
                model=model, 
                processor=processor, 
                conversations=batch_convs, 
                config=llm_eos_config,
                normalize=True, # Usually desirable for metric learning,
                intermediate_reasoning_drop_probability = 0.3
            )
            preds, vecs = head(vecs_o)
            ce_loss = torch.nn.functional.cross_entropy(preds, batch_labels)
            # 2. Compute Loss
            loss = loss_func(vecs, batch_labels) + 0.3 * ce_loss

        # Normalize loss for gradient accumulation
        loss = loss / accumulation_steps

        # 3. Backprop
        loss.backward()

        # Step and reset gradients only after accumulating enough steps
        if ((i + 1) % accumulation_steps == 0) or (i + 1 == len(trainloader)):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        real_loss = loss.item() * accumulation_steps
        epoch_loss += real_loss
        progress.set_postfix({"loss": f"{real_loss:.4f}"})

        writer.add_scalar('Loss/train_step', real_loss, global_step)
        global_step += 1
        # if i == 2:
        #     break
        # Free batch activations incrementally
        del vecs, loss, batch_convs, batch_labels

    avg_train_loss = epoch_loss / len(trainloader)
    writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
    print(f"Epoch {epoch+1} Complete | Average Train Loss: {avg_train_loss:.4f}")

    # Validation Loop (using k-NN classifier instead of ProxyAnchorLoss)
    model.eval()

    train_embeddings = []
    train_targets = []
    val_embeddings = []
    val_targets = []
    val_preds = []

    with torch.no_grad():
        # 1. Extract Train Embeddings
        i = 1
        train_progress = tqdm(trainloader, desc="Extracting Train Embeddings")
        for batch_convs, batch_labels in train_progress:
            with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16):
                vecs, lens = get_batch_conversation_embeddings_with_config(
                    model=model, 
                    processor=processor, 
                    conversations=batch_convs, 
                    config=llm_eos_config,
                    normalize=True
                )
                preds, vecs = head(vecs)
            train_embeddings.append(vecs.float().cpu())
            train_targets.append(batch_labels.cpu())
            del vecs, batch_convs, batch_labels, preds
            i += 1
            if i == 30:
                break

        train_embeddings_tensor = torch.cat(train_embeddings)
        train_targets_tensor = torch.cat(train_targets)
        
        # 2. Extract Val Embeddings
        val_progress = tqdm(valloader, desc="Extracting Val Embeddings")
        for batch_convs, batch_labels in val_progress:
            with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16):
                vecs, lens = get_batch_conversation_embeddings_with_config(
                    model=model, 
                    processor=processor, 
                    conversations=batch_convs, 
                    config=llm_eos_config,
                    normalize=True
                )
                preds, vecs = head(vecs)
            val_embeddings.append(vecs.float().cpu())
            val_targets.append(batch_labels.cpu())
            val_preds.append(preds.float().cpu())
            del vecs, batch_convs, batch_labels

        val_embeddings_tensor = torch.cat(val_embeddings)
        val_targets_tensor = torch.cat(val_targets)
        val_preds_tensor = torch.cat(val_preds)


    predicted_labels = val_preds_tensor.argmax(dim=1)
    head_acc = (predicted_labels == val_targets_tensor).float().mean().item()
    
    writer.add_scalar('Acc/val_head', head_acc, epoch)
    print(f"Epoch {epoch+1} Complete | Head Classification Val Accuracy: {head_acc:.4f}")

    # Fit k-NN and Score
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_embeddings_tensor.numpy(), train_targets_tensor.numpy())
    val_acc = knn.score(val_embeddings_tensor.numpy(), val_targets_tensor.numpy())

    writer.add_scalar('Acc/val_knn', val_acc, epoch)
    print(f"Epoch {epoch+1} Complete | k-NN Val Accuracy: {val_acc:.4f}")

    model.train()

    # --- Checkpointing ---
    if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == epochs:
        ckpt_dir = os.path.join(run_dir, f"checkpoint-{epoch + 1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Saving checkpoint to {ckpt_dir}...")

        model.save_pretrained(ckpt_dir)
        torch.save(head.state_dict(), os.path.join(ckpt_dir, "mc_head.pth"))
        torch.save(loss_func.state_dict(), os.path.join(ckpt_dir, "proxy_anchor_loss.pth"))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pth"))
        with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
            json.dump({"epoch": epoch + 1, "global_step": global_step}, f)
        print("Checkpoint saved successfully.")

model.save_pretrained("./test20poch")
writer.close()

print("Training cycle complete!")