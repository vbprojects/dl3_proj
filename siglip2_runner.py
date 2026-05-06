"""Training loop for SigLIP 2 fine-tuning with MissConfig PEFT.

Supports three training modes:
  - proxy_anchor: ProxyAnchorLoss on projected embeddings
  - joint: ProxyAnchorLoss + cross-entropy classification head
  - classification: Cross-entropy only

Evaluates with KNN and linear probe heads.
"""

import argparse
import csv
import json
import os

import pandas as pd
import torch
from dotenv import load_dotenv
from torch.optim import NAdam
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pytorch_metric_learning import losses
from sklearn.model_selection import train_test_split

from lvm_utils.mc_head import MonteCarloDropoutHead
from lvm_utils.model_helpers import (
    SIGLIP2_TARGET_MODULES,
    get_siglip2_image_embeddings,
    load_siglip2_model,
)
from peft import MissConfig, TaskType
from lvm_utils.utils import materialize_conversation_images
from lvm_utils.classification_heads import ClassificationHeadEvaluator


# ---------------------------------------------------------------------------
# Dataset (reuses existing cached conversation format, extracts images only)
# ---------------------------------------------------------------------------

class VLMDataset(Dataset):
    """Loads cached conversations and yields (image, label) pairs.

    Reuses the existing conversation cache format but only extracts the
    first image from each conversation for SigLIP 2 embedding extraction.
    """

    def __init__(self, index_df, cache_dir="./cache_data", label_mapping=None):
        self.df = index_df[index_df["status"].isin(["done", "STATUS_DONE"])].reset_index(
            drop=True
        )
        self.cache_dir = cache_dir

        if "label" not in self.df.columns:
            raise KeyError("The index dataframe must contain a 'label' column.")

        if label_mapping is None:
            self.labels, self.label_uniques = pd.factorize(self.df["label"])
            self.label_mapping = {val: i for i, val in enumerate(self.label_uniques)}
        else:
            self.label_mapping = label_mapping
            mapped_labels = self.df["label"].map(self.label_mapping)
            unknown_mask = mapped_labels.isna()
            if unknown_mask.any():
                self.df = self.df.loc[~unknown_mask].reset_index(drop=True)
                mapped_labels = mapped_labels.loc[~unknown_mask].reset_index(drop=True)
            self.labels = mapped_labels.astype("int64").to_numpy()
            self.label_uniques = pd.Series(list(self.label_mapping.keys()))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        conv_path = os.path.join(self.cache_dir, row["conversation_json_path"])
        with open(conv_path, "r") as f:
            raw_conv = json.load(f)

        conv = materialize_conversation_images(raw_conv, self.cache_dir)

        # Extract the first image from the conversation
        image = None
        for msg in conv:
            for item in msg.get("content", []):
                if item.get("type") == "image":
                    image = item["image"]
                    break
            if image is not None:
                break

        if image is None:
            raise ValueError(f"No image found in conversation at index {idx}")

        return image, self.labels[idx]


def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


# ---------------------------------------------------------------------------
# OOM recovery helpers
# ---------------------------------------------------------------------------

def _determine_max_batch_size_siglip(
    model, processor, dataset, head, loss_func,
    max_bs=32, training_mode="proxy_anchor", ce_loss_weight=0.3,
    device=None
):
    """Step-down search for the largest batch size that doesn't OOM."""
    if device is None:
        device = next(model.parameters()).device

    sample_size = min(max_bs, len(dataset))
    sample_images = []
    for i in range(sample_size):
        img, _ = dataset[i]
        sample_images.append(img)

    bs = max_bs
    while bs > 0:
        try:
            torch.cuda.empty_cache()
            test_images = sample_images[:bs]
            test_labels = torch.tensor(
                [dataset[i][1] for i in range(bs)], dtype=torch.long, device=device
            )

            model.eval()
            head.eval()
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    vecs_o = get_siglip2_image_embeddings(
                        model=model, processor=processor,
                        images=test_images, normalize=True,
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
            return bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
        except Exception:
            raise
    return 1


def _siglip_train_step_with_oom_recovery(
    model, processor, head, loss_func, batch_images, batch_labels,
    accumulation_steps, training_mode, ce_loss_weight, device
):
    """Process a SigLIP training batch, recursively splitting on OOM."""
    try:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            vecs_o = get_siglip2_image_embeddings(
                model=model, processor=processor,
                images=batch_images, normalize=True,
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
        mid = len(batch_images) // 2
        if mid < 1:
            print("  WARNING: Single-sample OOM, skipping batch.")
            return 0.0

        left_labels = batch_labels[:mid]
        right_labels = batch_labels[mid:]
        left_loss = _siglip_train_step_with_oom_recovery(
            model, processor, head, loss_func,
            batch_images[:mid], left_labels,
            accumulation_steps, training_mode, ce_loss_weight, device,
        )
        right_loss = _siglip_train_step_with_oom_recovery(
            model, processor, head, loss_func,
            batch_images[mid:], right_labels,
            accumulation_steps, training_mode, ce_loss_weight, device,
        )
        return (left_loss + right_loss) / 2


def _siglip_extract_batch_with_oom_recovery(
    model, processor, batch_images, head, device
):
    """Extract SigLIP embeddings from a batch, recursively splitting on OOM."""
    try:
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                vecs_o = get_siglip2_image_embeddings(
                    model=model, processor=processor,
                    images=batch_images, normalize=True,
                )
                return head(vecs_o)  # (preds, vecs)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        mid = len(batch_images) // 2
        if mid < 1:
            raise RuntimeError("Single-sample OOM during SigLIP extraction.")
        left = _siglip_extract_batch_with_oom_recovery(
            model, processor, batch_images[:mid], head, device
        )
        right = _siglip_extract_batch_with_oom_recovery(
            model, processor, batch_images[mid:], head, device
        )
        return (left[0] + right[0]) / 2, (left[1] + right[1]) / 2


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run SigLIP 2 Training with MissConfig PEFT using a config file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the JSON configuration file."
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    load_dotenv()

    # --resume_from flag takes precedence over config value
    resume_from = args.resume_from or config.get("resume_from_checkpoint", None)

    experiment_name = config.get("experiment_name", "siglip2_experiment")
    run_dir = os.path.join("runs", experiment_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    metrics_csv_path = os.path.join(run_dir, "metrics.csv")
    if not resume_from:
        with open(metrics_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_knn_acc", "val_linear_acc", "head_acc"])

    print(f"Starting SigLIP 2 experiment: {experiment_name}")

    # Build MiSS adapter config from config file
    peft_config = MissConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=config.get("miss_r", 16),
        mini_r=config.get("miss_mini_r", 8),
        miss_dropout=config.get("miss_dropout", 0.05),
        bias="none",
        target_modules=SIGLIP2_TARGET_MODULES,
    )

    # --- Load SigLIP 2 model ---
    model, processor = load_siglip2_model(peft_config=peft_config, load_peft=True)
    model.gradient_checkpointing_enable()

    # --- Load Data ---
    index_path = config.get("index_path", "cache_data/index.parquet")
    index = pd.read_parquet(index_path)

    train_index, val_index = train_test_split(
        index,
        test_size=config.get("test_size", 0.1),
        random_state=42,
        stratify=index["label"],
    )

    cache_dir = config.get("cache_dir", "./cache_data")
    train_dataset = VLMDataset(train_index, cache_dir=cache_dir)
    val_dataset = VLMDataset(
        val_index, cache_dir=cache_dir, label_mapping=train_dataset.label_mapping
    )

    base_batch_size = config.get("batch_size", 8)
    auto_batch = config.get("auto_batch_size", False)
    batch_size = base_batch_size

    # Auto-batch requires head + loss to exist first, so we create them
    # temporarily (same pattern as train_runner.py).
    # We will defer auto-batch logic to after head/loss setup.
    if auto_batch:
        batch_size = 1  # placeholder, will be replaced after head/loss setup

    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    valloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # --- Setup head ---
    try:
        embedding_size = model.config.vision_config.hidden_size
    except AttributeError:
        embedding_size = 768  # SigLIP base fallback

    output_embedding_size = config.get("output_embedding_size", 368)
    num_classes = len(train_dataset.label_uniques)

    print(
        f"Embedding dim: {embedding_size}, "
        f"Output dim: {output_embedding_size}, "
        f"Classes: {num_classes}"
    )

    head = MonteCarloDropoutHead(
        embedding_size,
        output_embedding_size,
        num_classes,
        dropout_prob=config.get("head_dropout_prob", 0.3),
    )
    head.to(model.device)

    # --- Loss function ---
    training_mode = config.get("training_mode", "proxy_anchor").lower()

    loss_func = None
    if training_mode in ("proxy_anchor", "joint"):
        loss_func = losses.ProxyAnchorLoss(
            num_classes=num_classes,
            embedding_size=output_embedding_size,
            margin=0.1,
            alpha=32,
        ).to(model.device)

    # Run auto-batch detection now that head + loss exist
    if auto_batch:
        batch_size = _determine_max_batch_size_siglip(
            model, processor, train_dataset, head, loss_func,
            max_bs=config.get("auto_batch_max_start", 32),
            training_mode=training_mode,
            ce_loss_weight=config.get("ce_loss_weight", 0.3),
        )
        torch.cuda.empty_cache()
        print(f"Auto-batch mode: determined max_batch_size = {batch_size}")
        # Rebuild DataLoaders with the detected batch size
        trainloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        valloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

    # --- Optimizer ---
    trainable_model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer_param_groups = [
        {
            "params": trainable_model_params,
            "lr": config.get("learning_rate_model", 5e-4),
        },
        {"params": head.parameters(), "lr": config.get("learning_rate_head", 5e-3)},
    ]
    if loss_func is not None:
        optimizer_param_groups.append(
            {
                "params": loss_func.parameters(),
                "lr": config.get("learning_rate_head", 5e-3),
            }
        )
    optimizer = NAdam(optimizer_param_groups)

    # --- Hyperparameters ---
    epochs = config.get("epochs", 20)
    accumulation_steps = config.get("accumulation_steps", 4)
    ce_loss_weight = config.get("ce_loss_weight", 0.3)
    checkpoint_interval = config.get("checkpoint_interval", 5)
    eval_interval = config.get("eval_interval", 1)

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

        # Restore training progress
        state_path = os.path.join(resume_from, "training_state.json")
        if os.path.isfile(state_path):
            with open(state_path, "r") as f:
                training_state = json.load(f)
            start_epoch = training_state.get("epoch", 0)
            global_step = training_state.get("global_step", 0)
            print(f"  Resuming from epoch {start_epoch}, global step {global_step}.")
        else:
            # Infer epoch from directory name (e.g. "checkpoint-10" → start at epoch 10)
            dir_name = os.path.basename(resume_from.rstrip("/"))
            if dir_name.startswith("checkpoint-"):
                try:
                    start_epoch = int(dir_name.split("-")[-1])
                    print(f"  Inferred start epoch {start_epoch} from directory name (no training_state.json found).")
                except ValueError:
                    pass

    # --- Evaluation ---
    head_evaluator = ClassificationHeadEvaluator(
        n_neighbors=config.get("k_neighbors", 5),
        linear_probe_max_iter=config.get("linear_probe_max_iter", 1000),
    )

    # --- TensorBoard ---
    tb_writer = SummaryWriter(log_dir=run_dir)

    print("Beginning SigLIP 2 Training Loop...")
    for epoch in range(start_epoch, epochs):
        model.train()
        head.train()
        torch.cuda.empty_cache()

        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        epoch_loss = 0.0

        progress = tqdm(trainloader, desc="Training")
        optimizer.zero_grad(set_to_none=True)

        for i, (batch_images, batch_labels) in enumerate(progress):
            batch_labels = batch_labels.to(model.device)

            real_loss = _siglip_train_step_with_oom_recovery(
                model, processor, head, loss_func,
                batch_images, batch_labels,
                accumulation_steps, training_mode, ce_loss_weight,
                model.device,
            )

            if real_loss == 0.0:
                continue

            if ((i + 1) % accumulation_steps == 0) or (i + 1 == len(trainloader)):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += real_loss
            progress.set_postfix({"loss": f"{real_loss:.4f}"})

            tb_writer.add_scalar("Loss/train_step", real_loss, global_step)
            global_step += 1

            del batch_images, batch_labels

        avg_train_loss = epoch_loss / len(trainloader)
        tb_writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        print(f"Epoch {epoch + 1} Complete | Average Train Loss: {avg_train_loss:.4f}")

        # --- Evaluation ---
        val_knn_acc = 0.0
        val_linear_acc = 0.0
        head_acc = 0.0

        if (epoch + 1) % eval_interval == 0:
            model.eval()
            head.eval()

            train_embeddings = []
            train_targets = []
            val_embeddings = []
            val_targets = []
            val_preds_list = []

            # Extract train embeddings
            train_progress = tqdm(trainloader, desc="Extracting Train Embeddings")
            for batch_images, batch_labels in train_progress:
                preds_out, vecs_out = _siglip_extract_batch_with_oom_recovery(
                    model, processor, batch_images, head, model.device
                )
                train_embeddings.append(vecs_out.float().cpu())
                train_targets.append(batch_labels.cpu())
                del preds_out, vecs_out, batch_images, batch_labels

            train_embeddings_tensor = torch.cat(train_embeddings)
            train_targets_tensor = torch.cat(train_targets)

            # Fit evaluator on train embeddings (once per eval)
            head_evaluator.fit_train(train_embeddings_tensor, train_targets_tensor)

            # Extract val embeddings
            val_progress = tqdm(valloader, desc="Extracting Val Embeddings")
            for batch_images, batch_labels in val_progress:
                preds_out, vecs_out = _siglip_extract_batch_with_oom_recovery(
                    model, processor, batch_images, head, model.device
                )
                val_embeddings.append(vecs_out.float().cpu())
                val_targets.append(batch_labels.cpu())
                val_preds_list.append(preds_out.float().cpu())
                del preds_out, vecs_out, batch_images, batch_labels

            val_embeddings_tensor = torch.cat(val_embeddings)
            val_targets_tensor = torch.cat(val_targets)
            val_preds_tensor = torch.cat(val_preds_list)

            # Head classification accuracy
            predicted_labels = val_preds_tensor.argmax(dim=1)
            head_acc = (
                (predicted_labels == val_targets_tensor).float().mean().item()
            )
            tb_writer.add_scalar("Acc/val_head", head_acc, epoch)
            print(
                f"Epoch {epoch + 1} | Head Classification Val Accuracy: {head_acc:.4f}"
            )

            # KNN evaluation
            val_knn_acc = head_evaluator.knn_accuracy(
                val_embeddings_tensor, val_targets_tensor
            )
            tb_writer.add_scalar("Acc/val_knn", val_knn_acc, epoch)
            print(f"Epoch {epoch + 1} | k-NN Val Accuracy: {val_knn_acc:.4f}")

            # Linear probe evaluation
            val_linear_acc = head_evaluator.linear_probe_accuracy(
                val_embeddings_tensor, val_targets_tensor
            )
            tb_writer.add_scalar("Acc/val_linear_probe", val_linear_acc, epoch)
            print(
                f"Epoch {epoch + 1} | Linear Probe Val Accuracy: {val_linear_acc:.4f}"
            )

        # --- Logging ---
        with open(metrics_csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch + 1, avg_train_loss, val_knn_acc, val_linear_acc, head_acc]
            )

        # --- Checkpointing ---
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == epochs:
            ckpt_dir = os.path.join(run_dir, f"checkpoint-{epoch + 1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            print(f"Saving checkpoint to {ckpt_dir}...")

            model.save_pretrained(ckpt_dir)
            torch.save(head.state_dict(), os.path.join(ckpt_dir, "mc_head.pth"))
            if loss_func is not None:
                torch.save(
                    loss_func.state_dict(),
                    os.path.join(ckpt_dir, "proxy_anchor_loss.pth"),
                )
            torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pth"))
            with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
                json.dump({"epoch": epoch + 1, "global_step": global_step}, f)
            print("Checkpoint saved successfully.")

    tb_writer.close()
    print("SigLIP 2 training completely finished!")


if __name__ == "__main__":
    main()