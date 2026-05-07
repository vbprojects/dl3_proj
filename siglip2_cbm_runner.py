"""Training loop for SigLIP 2 fine-ting with InfoNCE + CrossBatchMemory.

Uses NTXentLoss (InfoNCE) from pytorch-metric-learning wrapped in CrossBatchMemory
to leverage a queue of embeddings from previous batches for contrastive learning.
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
    """Loads cached conversations and yields (image, label) pairs."""

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
    max_bs=32, training_mode="ntxent_cbm",
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
                    _, vecs = head(vecs_o)
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
    accumulation_steps, device
):
    """Process a SigLIP training batch, recursively splitting on OOM."""
    try:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            vecs_o = get_siglip2_image_embeddings(
                model=model, processor=processor,
                images=batch_images, normalize=True,
            )
            _, vecs = head(vecs_o)
            loss = loss_func(vecs.float(), batch_labels)

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
            accumulation_steps, device,
        )
        right_loss = _siglip_train_step_with_oom_recovery(
            model, processor, head, loss_func,
            batch_images[mid:], right_labels,
            accumulation_steps, device,
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
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run SigLIP 2 Training with InfoNCE + CrossBatchMemory."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the JSON configuration file."
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    load_dotenv()

    experiment_name = config.get("experiment_name", "siglip2_ntxent_cbm")
    run_dir = os.path.join("runs", experiment_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    metrics_csv_path = os.path.join(run_dir, "metrics.csv")
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
    index_path = config.get("index_path", "cached_cifar100/index.parquet")
    index = pd.read_parquet(index_path)

    train_index, val_index = train_test_split(
        index,
        test_size=config.get("test_size", 0.1),
        random_state=42,
        stratify=index["label"],
    )

    # Optional: sample a fraction of the training data
    train_fraction = config.get("train_fraction", 1.0)
    if train_fraction < 1.0:
        train_index = train_index.sample(
            frac=train_fraction,
            random_state=42,
            replace=False,
        )
        print(f"Sampled {train_fraction*100:.1f}% of training data: {len(train_index)} samples (from {len(index)} total)")

    cache_dir = config.get("cache_dir", "./cached_cifar100")
    train_dataset = VLMDataset(train_index, cache_dir=cache_dir)
    val_dataset = VLMDataset(
        val_index, cache_dir=cache_dir, label_mapping=train_dataset.label_mapping
    )

    base_batch_size = config.get("batch_size", 8)
    auto_batch = config.get("auto_batch_size", False)
    batch_size = base_batch_size

    # Placeholder - will be set after head/loss setup if auto_batch
    if auto_batch:
        batch_size = 1

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

    # --- Loss function: InfoNCE (NTXentLoss) + CrossBatchMemory ---
    temperature = config.get("temperature", 0.07)
    memory_size = config.get("memory_size", 1024)

    print(f"Using NTXentLoss (InfoNCE) with temperature={temperature}, wrapped in CrossBatchMemory (memory_size={memory_size})")

    base_loss = losses.NTXentLoss(temperature=temperature)
    loss_func = losses.CrossBatchMemory(
        loss=base_loss,
        embedding_size=output_embedding_size,
        memory_size=memory_size,
    ).to(model.device)

    # Run auto-batch detection now that head + loss exist
    if auto_batch:
        batch_size = _determine_max_batch_size_siglip(
            model, processor, train_dataset, head, loss_func,
            max_bs=config.get("auto_batch_max_start", 32),
            training_mode="ntxent_cbm",
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
    # Note: NTXentLoss + CrossBatchMemory do NOT have learnable parameters,
    # so we do not add loss_func.parameters() to the optimizer.

    optimizer = NAdam(optimizer_param_groups)

    # --- Hyperparameters ---
    epochs = config.get("epochs", 20)
    accumulation_steps = config.get("accumulation_steps", 4)
    checkpoint_interval = config.get("checkpoint_interval", 5)
    eval_interval = config.get("eval_interval", 1)

    # --- Evaluation ---
    head_evaluator = ClassificationHeadEvaluator(
        n_neighbors=config.get("k_neighbors", 5),
        linear_probe_max_iter=config.get("linear_probe_max_iter", 1000),
    )

    # --- TensorBoard ---
    tb_writer = SummaryWriter(log_dir=run_dir)

    print("Beginning SigLIP 2 Training Loop with InfoNCE + CrossBatchMemory...")
    global_step = 0

    for epoch in range(epochs):
        model.train()
        head.train()
        # Reset the CrossBatchMemory queue at the start of each epoch
        loss_func.reset_queue()
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
                accumulation_steps,
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

            # Optional: sample a subset of training data for evaluation
            eval_train_fraction = config.get("eval_train_fraction", 1.0)
            eval_train_dataset = train_dataset
            if eval_train_fraction < 1.0:
                num_samples = max(1, int(len(train_dataset) * eval_train_fraction))
                indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(42))[:num_samples]
                from torch.utils.data import Subset
                eval_train_dataset = Subset(train_dataset, indices.tolist())
                print(f"  Sampling {num_samples}/{len(train_dataset)} training examples for evaluation ({eval_train_fraction*100:.1f}%)")

            eval_trainloader = DataLoader(
                eval_train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
            )

            # Extract train embeddings
            train_progress = tqdm(eval_trainloader, desc="Extracting Train Embeddings")
            for batch_images, batch_labels in train_progress:
                preds_out, vecs_out = _siglip_extract_batch_with_oom_recovery(
                    model, processor, batch_images, head, model.device
                )
                train_embeddings.append(vecs_out.float().cpu())
                train_targets.append(batch_labels.cpu())
                del preds_out, vecs_out, batch_images, batch_labels

            train_embeddings_tensor = torch.cat(train_embeddings)
            train_targets_tensor = torch.cat(train_targets)

            # Fit evaluator on train embeddings
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
            torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pth"))
            print("Checkpoint saved successfully.")

    tb_writer.close()
    print("SigLIP 2 training with InfoNCE + CrossBatchMemory completely finished!")


if __name__ == "__main__":
    main()