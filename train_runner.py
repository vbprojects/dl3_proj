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
from lvm_utils.model_helpers import load_model_id, build_conversation_embedding_config, get_batch_conversation_embeddings_with_config, target_modules
from lvm_utils.mc_head import MonteCarloDropoutHead
from peft import MissConfig, TaskType

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


def _determine_max_batch_size(
    model, processor, dataset, head, loss_func, llm_eos_config,
    max_bs=32, drop_prob=0.0, training_mode="proxy_anchor", ce_loss_weight=0.3,
    device=None
):
    """Binary-search / step-down for the largest batch size that doesn't OOM."""
    if device is None:
        device = next(model.parameters()).device

    # grab a representative sample of `max_bs` items (or fewer)
    sample_size = min(max_bs, len(dataset))
    sample_convs = []
    for i in range(sample_size):
        conv, _ = dataset[i]
        sample_convs.append(conv)

    bs = max_bs
    while bs > 0:
        try:
            torch.cuda.empty_cache()
            test_convs = sample_convs[:bs]
            test_labels = torch.tensor(
                [dataset[i][1] for i in range(bs)], dtype=torch.long, device=device
            )

            model.eval()
            head.eval()
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    vecs_o, _ = get_batch_conversation_embeddings_with_config(
                        model=model,
                        processor=processor,
                        conversations=test_convs,
                        config=llm_eos_config,
                        normalize=True,
                        intermediate_reasoning_drop_probability=drop_prob,
                    )
                    preds, vecs = head(vecs_o)
                    if training_mode == "joint":
                        ce_loss = torch.nn.functional.cross_entropy(preds, test_labels)
                        _ = loss_func(vecs, test_labels) + ce_loss_weight * ce_loss
                    elif training_mode == "classification":
                        _ = torch.nn.functional.cross_entropy(preds, test_labels)
                    else:
                        _ = loss_func(vecs, test_labels)
            model.train()
            head.train()
            # Success – this batch size works
            return bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
        except Exception:
            # Non-memory error – re-raise
            raise
    return 1


def _train_step_with_oom_recovery(
    model, processor, head, loss_func, batch_convs, batch_labels,
    llm_eos_config, accumulation_steps, training_mode, ce_loss_weight,
    drop_prob, device
):
    """Process a training batch, recursively splitting on OOM."""
    try:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            vecs_o, lens = get_batch_conversation_embeddings_with_config(
                model=model,
                processor=processor,
                conversations=batch_convs,
                config=llm_eos_config,
                normalize=True,
                intermediate_reasoning_drop_probability=drop_prob,
            )
            preds, vecs = head(vecs_o)

            if training_mode == "joint":
                ce_loss = torch.nn.functional.cross_entropy(preds, batch_labels)
                loss = loss_func(vecs, batch_labels) + ce_loss_weight * ce_loss
            elif training_mode == "classification":
                loss = torch.nn.functional.cross_entropy(preds, batch_labels)
            else:
                loss = loss_func(vecs, batch_labels)

        loss = loss / accumulation_steps
        loss.backward()
        return loss.item() * accumulation_steps
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        mid = len(batch_convs) // 2
        if mid < 1:
            # Single sample still OOMs – skip this batch
            print("  WARNING: Single-sample OOM, skipping batch.")
            return 0.0

        left_labels = batch_labels[:mid]
        right_labels = batch_labels[mid:]
        left_loss = _train_step_with_oom_recovery(
            model, processor, head, loss_func,
            batch_convs[:mid], left_labels,
            llm_eos_config, accumulation_steps, training_mode, ce_loss_weight,
            drop_prob, device,
        )
        right_loss = _train_step_with_oom_recovery(
            model, processor, head, loss_func,
            batch_convs[mid:], right_labels,
            llm_eos_config, accumulation_steps, training_mode, ce_loss_weight,
            drop_prob, device,
        )
        return (left_loss + right_loss) / 2


def _extract_batch_embeddings_with_oom_recovery(
    model, processor, batch_convs, head, llm_eos_config, device
):
    """Extract embeddings from a batch, recursively splitting on OOM."""
    try:
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                vecs_o, _ = get_batch_conversation_embeddings_with_config(
                    model=model,
                    processor=processor,
                    conversations=batch_convs,
                    config=llm_eos_config,
                    normalize=True,
                )
                return head(vecs_o)[0]  # return (preds, vecs) – caller can pick
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        mid = len(batch_convs) // 2
        if mid < 1:
            raise RuntimeError("Single-sample OOM during extraction.")
        left = _extract_batch_embeddings_with_oom_recovery(
            model, processor, batch_convs[:mid], head, llm_eos_config, device
        )
        right = _extract_batch_embeddings_with_oom_recovery(
            model, processor, batch_convs[mid:], head, llm_eos_config, device
        )
        return (left + right) / 2


def _resolve_checkpoint_dir(path: str) -> str:
    """Resolve a path to the actual checkpoint directory containing adapter weights.

    Accepts either:
    - A direct checkpoint directory (``checkpoint-N/``) containing adapter weights.
    - A run directory whose subdirectories contain ``checkpoint-N/`` folders;
      in this case the subdirectory with the highest N is selected automatically.

    Raises FileNotFoundError if no valid checkpoint can be found.
    """
    # Direct checkpoint directory: already contains adapter weights
    if (os.path.isfile(os.path.join(path, "adapter_model.safetensors")) or
            os.path.isfile(os.path.join(path, "adapter_model.bin"))):
        return path

    # Run directory: look for checkpoint-N subdirectories
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

    # Pick the checkpoint with the highest epoch number
    candidates.sort(key=lambda x: x[0])
    latest_epoch, latest_path = candidates[-1]
    print(f"  Auto-selected latest checkpoint: {latest_path} (epoch {latest_epoch})")
    return latest_path


def _infer_start_epoch(resume_dir: str):
    """Infer the start epoch and global step from a checkpoint directory.

    Resolution order:
    1. ``training_state.json`` written by the runner at save time.
    2. Parse the epoch number from a directory name matching ``checkpoint-{N}``.
    3. Return (0, 0) if neither source is available.

    Returns:
        (start_epoch, global_step)
    """
    state_path = os.path.join(resume_dir, "training_state.json")
    if os.path.isfile(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
        return state.get("epoch", 0), state.get("global_step", 0)

    # Fall back to parsing the directory name (e.g. "checkpoint-10")
    dir_name = os.path.basename(resume_dir.rstrip("/"))
    if dir_name.startswith("checkpoint-"):
        try:
            epoch = int(dir_name.split("-")[-1])
            print(f"  Inferred start epoch {epoch} from directory name (no training_state.json found).")
            return epoch, 0
        except ValueError:
            pass
    return 0, 0


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


def main():
    parser = argparse.ArgumentParser(description="Run LVM Training with PEFT using a config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from.",
    )
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        config = json.load(f)

    load_dotenv()

    # --resume_from flag takes precedence over config value
    resume_from = args.resume_from or config.get("resume_from_checkpoint", None)

    experiment_name = config.get("experiment_name", "experiment")
    run_dir = os.path.join("runs", experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save a copy of the config in the run directory for reproducibility
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
        
    metrics_csv_path = os.path.join(run_dir, "metrics.csv")
    if not resume_from:
        with open(metrics_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_acc", "head_acc"])

    print(f"Starting experiment: {experiment_name}")

    # Build MiSS adapter config from config file
    peft_config = MissConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.get("miss_r", 16),
        mini_r=config.get("miss_mini_r", 8),
        miss_dropout=config.get("miss_dropout", 0.05),
        bias="none",
        target_modules=target_modules,
    )

    # Load Model
    model, processor = load_model_id(peft_config=peft_config, load_peft=True)
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

    # Determine batch size (auto-detect if enabled in config)
    base_batch_size = config.get("batch_size", 5)
    auto_batch = config.get("auto_batch_size", False)
    batch_size = base_batch_size

    if auto_batch:
        # Temp head/loss for probing (need them on device before auto-batch)
        print("Auto-batch mode: setting up temporary head and loss for probing...")
        try:
            _embed = model.config.text_config.hidden_size
        except AttributeError:
            _embed = 1024
        _num_classes = len(train_dataset.label_uniques)
        _temp_head = MonteCarloDropoutHead(
            _embed, config.get("output_embedding_size", 368), _num_classes,
            dropout_prob=config.get("head_dropout_prob", 0.3)
        ).to(model.device)
        _temp_loss = None
        _mode = config.get("training_mode", "proxy_anchor").lower()
        if _mode in ("proxy_anchor", "joint"):
            _temp_loss = losses.ProxyAnchorLoss(
                num_classes=_num_classes,
                embedding_size=config.get("output_embedding_size", 368),
                margin=0.1, alpha=32,
            ).to(model.device)
        _llm_cfg = build_conversation_embedding_config(processor)
        batch_size = _determine_max_batch_size(
            model, processor, train_dataset, _temp_head, _temp_loss, _llm_cfg,
            max_bs=config.get("auto_batch_max_start", 32),
            drop_prob=config.get("intermediate_reasoning_drop_probability", 0.3),
            training_mode=_mode,
            ce_loss_weight=config.get("ce_loss_weight", 0.3),
        )
        del _temp_head, _temp_loss, _llm_cfg
        torch.cuda.empty_cache()
        print(f"Auto-batch mode: determined max_batch_size = {batch_size}")

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

    # --- Resume from checkpoint ---
    start_epoch = 0
    global_step = 0

    if resume_from:
        resume_from = _resolve_checkpoint_dir(resume_from)
        print(f"Resuming training from checkpoint: {resume_from}")

        # Restore PEFT adapter weights
        _load_peft_adapter(model, resume_from)
        print("  Loaded PEFT adapter weights.")

        # Restore head
        head_ckpt = os.path.join(resume_from, "mc_head.pth")
        if os.path.isfile(head_ckpt):
            head.load_state_dict(torch.load(head_ckpt, map_location=model.device))
            print("  Loaded MC head weights.")

        # Restore loss function proxies
        loss_ckpt = os.path.join(resume_from, "proxy_anchor_loss.pth")
        if loss_func is not None and os.path.isfile(loss_ckpt):
            loss_func.load_state_dict(torch.load(loss_ckpt, map_location=model.device))
            print("  Loaded ProxyAnchorLoss weights.")

        # Restore optimizer
        opt_ckpt = os.path.join(resume_from, "optimizer.pth")
        if os.path.isfile(opt_ckpt):
            optimizer.load_state_dict(torch.load(opt_ckpt, map_location="cpu"))
            print("  Loaded optimizer state.")

        # Restore training progress (with automatic inference from dir name as fallback)
        start_epoch, global_step = _infer_start_epoch(resume_from)
        print(f"  Resuming from epoch {start_epoch}, global step {global_step}.")

    tb_writer = SummaryWriter(log_dir=run_dir)

    print("Beginning Training Loop...")
    for epoch in range(start_epoch, epochs):
        model.train()
        head.train()
        torch.cuda.empty_cache()

        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        epoch_loss = 0.0
        
        progress = tqdm(trainloader, desc="Training")
        optimizer.zero_grad(set_to_none=True)
        
        for i, (batch_convs, batch_labels) in enumerate(progress):
            batch_labels = batch_labels.to(model.device)

            real_loss = _train_step_with_oom_recovery(
                model, processor, head, loss_func,
                batch_convs, batch_labels,
                llm_eos_config, accumulation_steps,
                training_mode, ce_loss_weight,
                intermediate_reasoning_drop_probability,
                model.device,
            )

            if real_loss == 0.0:
                # Batch was skipped due to single-sample OOM
                continue

            if ((i + 1) % accumulation_steps == 0) or (i + 1 == len(trainloader)):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += real_loss
            progress.set_postfix({"loss": f"{real_loss:.4f}"})

            tb_writer.add_scalar('Loss/train_step', real_loss, global_step)
            global_step += 1

            del batch_convs, batch_labels
        
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
            
            train_progress = tqdm(trainloader, desc="Extracting Train Embeddings")
            for batch_convs, batch_labels in train_progress:
                preds_all = _extract_batch_embeddings_with_oom_recovery(
                    model, processor, batch_convs, head, llm_eos_config, model.device
                )
                train_embeddings.append(preds_all[1].float().cpu())
                train_targets.append(batch_labels.cpu())
                del preds_all, batch_convs, batch_labels

            train_embeddings_tensor = torch.cat(train_embeddings)
            train_targets_tensor = torch.cat(train_targets)

            val_preds_list = []
            val_embeddings = []
            val_targets = []
            val_progress = tqdm(valloader, desc="Extracting Val Embeddings")
            for batch_convs, batch_labels in val_progress:
                preds_all = _extract_batch_embeddings_with_oom_recovery(
                    model, processor, batch_convs, head, llm_eos_config, model.device
                )
                val_embeddings.append(preds_all[1].float().cpu())
                val_targets.append(batch_labels.cpu())
                val_preds_list.append(preds_all[0].float().cpu())
                del preds_all, batch_convs, batch_labels

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
            torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pth"))
            with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
                json.dump({"epoch": epoch + 1, "global_step": global_step}, f)

            print(f"Checkpoint saved successfully.")

    tb_writer.close()
    print("Training completely finished!")

if __name__ == "__main__":
    main()
