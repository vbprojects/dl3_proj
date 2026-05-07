"""Evaluate a trained model using Monte-Carlo Dropout with KNN and Linear Probe heads.

Supports both SigLIP 2 and LVM models. Loads the latest checkpoint from a run
directory, trains KNN and Linear Probe classifiers on training embeddings, then
evaluates on a separate test set using Monte-Carlo dropout sampling.

Usage examples:
    # Evaluate a SigLIP 2 run
    python evaluate_mc_dropout.py \
        --run_dir runs/cifar100_ntxent_cbm \
        --model_type siglip \
        --train_cache_dir cached_cifar100 \
        --test_cache_dir cached_cifar100_test \
        --mc_samples 20 \
        --batch_size 512 

    # Evaluate an LVM run
    python evaluate_mc_dropout.py \
        --run_dir runs/cifar100_miss_medium \
        --model_type lvm \
        --train_cache_dir cached_cifar100 \
        --test_cache_dir cached_cifar100_test \
        --mc_samples 20 \
        --batch_size 128
"""

import argparse
import csv
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from peft import MissConfig, TaskType, set_peft_model_state_dict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lvm_utils.mc_head import MonteCarloDropoutHead


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class LVMVLMDataset(Dataset):
    """Loads cached conversations and yields (conversation, label) pairs for LVM."""

    def __init__(self, index_df, cache_dir="./cache_data", label_mapping=None):
        from lvm_utils.utils import materialize_conversation_images

        self.df = index_df[index_df["status"].isin(["done", "STATUS_DONE"])].reset_index(
            drop=True
        )
        self.cache_dir = cache_dir
        self._materialize = materialize_conversation_images

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
        conv = self._materialize(raw_conv, self.cache_dir)
        return conv, self.labels[idx]


class SigLIP2VLMDataset(Dataset):
    """Loads cached conversations and yields (image, label) pairs for SigLIP 2."""

    def __init__(self, index_df, cache_dir="./cache_data", label_mapping=None):
        from lvm_utils.utils import materialize_conversation_images

        self.df = index_df[index_df["status"].isin(["done", "STATUS_DONE"])].reset_index(
            drop=True
        )
        self.cache_dir = cache_dir
        self._materialize = materialize_conversation_images

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
        conv = self._materialize(raw_conv, self.cache_dir)

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


def _lvm_collate_fn(batch):
    convs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return convs, labels


def _siglip_collate_fn(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _resolve_checkpoint_dir(path: str) -> str:
    """Resolve path to the latest checkpoint directory."""
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
            "Provide a path to a specific 'checkpoint-N' directory or a run directory."
        )

    candidates.sort(key=lambda x: x[0])
    latest_epoch, latest_path = candidates[-1]
    print(f"  Auto-selected latest checkpoint: {latest_path} (epoch {latest_epoch})")
    return latest_path


def _load_peft_adapter(model, ckpt_dir: str):
    """Load PEFT adapter weights from checkpoint directory."""
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
# OOM-safe embedding extraction
# ---------------------------------------------------------------------------

def _lvm_extract_batch_embeddings(
    model, processor, batch_convs, head, llm_eos_config, device
):
    """Extract LVM embeddings from a batch, recursively splitting on OOM."""
    from lvm_utils.model_helpers import get_batch_conversation_embeddings_with_config

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
                return head(vecs_o)  # (preds, vecs)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        mid = len(batch_convs) // 2
        if mid < 1:
            raise RuntimeError("Single-sample OOM during LVM extraction.")
        left = _lvm_extract_batch_embeddings(
            model, processor, batch_convs[:mid], head, llm_eos_config, device
        )
        right = _lvm_extract_batch_embeddings(
            model, processor, batch_convs[mid:], head, llm_eos_config, device
        )
        return torch.cat([left[0], right[0]], dim=0), torch.cat([left[1], right[1]], dim=0)


def _siglip_extract_batch_embeddings(
    model, processor, batch_images, head, device
):
    """Extract SigLIP embeddings from a batch, recursively splitting on OOM."""
    from lvm_utils.model_helpers import get_siglip2_image_embeddings

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
        left = _siglip_extract_batch_embeddings(
            model, processor, batch_images[:mid], head, device
        )
        right = _siglip_extract_batch_embeddings(
            model, processor, batch_images[mid:], head, device
        )
        return torch.cat([left[0], right[0]], dim=0), torch.cat([left[1], right[1]], dim=0)


# ---------------------------------------------------------------------------
# MC Dropout evaluation helpers
# ---------------------------------------------------------------------------

def _top_k_accuracy(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Calculate top-k accuracy from probability predictions and true labels.
    
    Args:
        probs: [N, num_classes] probability predictions
        labels: [N] true labels
        k: Top-k value
        
    Returns:
        Top-k accuracy as a float between 0 and 1.
    """
    N = len(labels)
    correct = 0
    for i in range(N):
        sorted_indices = np.argsort(probs[i])[::-1]
        if labels[i] in sorted_indices[:k]:
            correct += 1
    return correct / N


def _knn_predict_proba_simple(
    knn: KNeighborsClassifier,
    val_embeddings: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Use the fitted KNN's predict_proba directly if it supports it."""
    # sklearn KNeighborsClassifier has predict_proba when metric supports it
    try:
        return knn.predict_proba(val_embeddings)
    except Exception:
        # Fallback: one-hot encode predictions
        preds = knn.predict(val_embeddings)
        probs = np.zeros((len(preds), num_classes), dtype=np.float64)
        for i, label in enumerate(preds):
            class_idx = list(knn.classes_).index(label)
            probs[i, class_idx] = 1.0
        return probs


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model using Monte-Carlo Dropout with KNN "
                    "and Linear Probe heads. Supports both SigLIP 2 and LVM models."
    )
    parser.add_argument(
        "--run_dir", type=str, required=True,
        help="Path to the run directory (e.g., runs/siglip2_proxy_anchor).",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, choices=["lvm", "siglip"],
        help="Model type: 'lvm' for LVM models, 'siglip' for SigLIP 2 models.",
    )
    parser.add_argument(
        "--train_cache_dir", type=str, required=True,
        help="Path to cached training dataset directory with index.parquet "
             "(e.g., cached_cifar100).",
    )
    parser.add_argument(
        "--test_cache_dir", type=str, required=True,
        help="Path to cached test dataset directory with index.parquet "
             "(e.g., cached_cifar100_test).",
    )
    parser.add_argument(
        "--mc_samples", type=int, default=20,
        help="Number of Monte-Carlo dropout samples (default: 10).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Evaluation batch size (default: 32).",
    )
    parser.add_argument(
        "--k_neighbors", type=int, default=None,
        help="Number of neighbors for KNN (default: from config or 5).",
    )
    parser.add_argument(
        "--linear_probe_max_iter", type=int, default=None,
        help="Max iterations for linear probe (default: from config or 1000).",
    )
    args = parser.parse_args()

    load_dotenv()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. Load config from run directory
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    config_path = os.path.join(args.run_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}. "
                                "Ensure --run_dir points to a valid run directory.")
    with open(config_path, "r") as f:
        config = json.load(f)

    k_neighbors = args.k_neighbors or config.get("k_neighbors", 5)
    linear_probe_max_iter = args.linear_probe_max_iter or config.get("linear_probe_max_iter", 1000)
    output_embedding_size = config.get("output_embedding_size", 368)
    head_dropout_prob = config.get("head_dropout_prob", 0.3)
    miss_r = config.get("miss_r", 16)
    miss_mini_r = config.get("miss_mini_r", 8)
    miss_dropout = config.get("miss_dropout", 0.05)
    train_fraction = config.get("train_fraction", 0.3)

    print(f"\n{'='*60}")
    print(f"Monte-Carlo Dropout Evaluation")
    print(f"{'='*60}")
    print(f"  Model type       : {args.model_type.upper()}")
    print(f"  Run directory    : {args.run_dir}")
    print(f"  Train cache dir  : {args.train_cache_dir}")
    print(f"  Test cache dir   : {args.test_cache_dir}")
    print(f"  MC samples       : {args.mc_samples}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  KNN neighbors    : {k_neighbors}")
    if train_fraction < 1.0:
        print(f"  Train fraction   : {train_fraction}")
    print(f"{'='*60}\n")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. Find and load the latest checkpoint
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ckpt_dir = _resolve_checkpoint_dir(args.run_dir)
    print(f"Loading checkpoint from: {ckpt_dir}\n")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. Initialize model + processor
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if args.model_type == "lvm":
        from lvm_utils.model_helpers import (
            load_model_id,
            build_conversation_embedding_config,
            get_batch_conversation_embeddings_with_config,
            target_modules,
        )

        peft_config = MissConfig(
            task_type=TaskType.CAUSAL_LM,
            r=miss_r,
            mini_r=miss_mini_r,
            miss_dropout=miss_dropout,
            bias="none",
            target_modules=target_modules,
        )
        model, processor = load_model_id(peft_config=peft_config, load_peft=True)
        model.gradient_checkpointing_enable()

        try:
            embedding_size = model.config.text_config.hidden_size
        except AttributeError:
            embedding_size = 1024

        llm_eos_config = build_conversation_embedding_config(processor)

    else:  # siglip
        from lvm_utils.model_helpers import (
            load_siglip2_model,
            get_siglip2_image_embeddings,
            SIGLIP2_TARGET_MODULES,
        )

        peft_config = MissConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=miss_r,
            mini_r=miss_mini_r,
            miss_dropout=miss_dropout,
            bias="none",
            target_modules=SIGLIP2_TARGET_MODULES,
        )
        model, processor = load_siglip2_model(peft_config=peft_config, load_peft=True)
        model.gradient_checkpointing_enable()

        try:
            embedding_size = model.config.vision_config.hidden_size
        except AttributeError:
            embedding_size = 768  # SigLIP base fallback

        llm_eos_config = None

    device = model.device
    print(f"  Embedding size   : {embedding_size}")
    print(f"  Device           : {device}\n")

    # Load PEFT adapter weights
    _load_peft_adapter(model, ckpt_dir)
    print("  Loaded PEFT adapter weights.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. Initialize and load head weights
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Determine num_classes from training dataset first (needed for head init)
    train_index_path = os.path.join(args.train_cache_dir, "index.parquet")
    if not os.path.isfile(train_index_path):
        raise FileNotFoundError(f"Train index not found at {train_index_path}")
    train_index_full = pd.read_parquet(train_index_path)

    # Optional: sample a fraction of the training data (e.g., 0.3 = 30%)
    # to match the training configuration used during model training
    if train_fraction < 1.0:
        original_train_size = len(train_index_full)
        train_index_full = train_index_full.sample(
            frac=train_fraction,
            random_state=42,
            replace=False,
        )
        print(f"  Sampled {train_fraction*100:.1f}% of training data: "
              f"{len(train_index_full)} samples (from {original_train_size} total)")

    _, train_label_uniques = pd.factorize(train_index_full["label"])
    num_classes = len(train_label_uniques)
    train_label_mapping = {val: i for i, val in enumerate(train_label_uniques)}

    head = MonteCarloDropoutHead(
        embedding_size,
        output_embedding_size,
        num_classes,
        dropout_prob=head_dropout_prob,
    )
    head.to(device)

    head_ckpt = os.path.join(ckpt_dir, "mc_head.pth")
    if os.path.isfile(head_ckpt):
        head.load_state_dict(torch.load(head_ckpt, map_location=device))
        print("  Loaded MC head weights.")
    else:
        print(f"  WARNING: Head checkpoint not found at {head_ckpt}, using random weights.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. Load datasets
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Training dataset (for extracting embeddings to train KNN/Linear Probe)
    if args.model_type == "lvm":
        train_dataset = LVMVLMDataset(train_index_full, cache_dir=args.train_cache_dir)
    else:
        train_dataset = SigLIP2VLMDataset(train_index_full, cache_dir=args.train_cache_dir)

    num_classes = len(train_dataset.label_uniques)
    label_mapping = train_dataset.label_mapping

    print(f"\n  Train samples    : {len(train_dataset)}")
    print(f"  Num classes      : {num_classes}")

    # Test dataset
    test_index_path = os.path.join(args.test_cache_dir, "index.parquet")
    if not os.path.isfile(test_index_path):
        raise FileNotFoundError(f"Test index not found at {test_index_path}")
    test_index = pd.read_parquet(test_index_path)

    if args.model_type == "lvm":
        # Filter test dataset to only include labels present in training set
        test_dataset = LVMVLMDataset(
            test_index, cache_dir=args.test_cache_dir,
            label_mapping=label_mapping
        )
        # Create dataloader with custom collate
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=_lvm_collate_fn
        )
        # Also need train_loader for LVM
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=_lvm_collate_fn
        )
    else:
        # Filter test dataset to only include labels present in training set
        test_dataset = SigLIP2VLMDataset(
            test_index, cache_dir=args.test_cache_dir,
            label_mapping=label_mapping
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=_siglip_collate_fn
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=_siglip_collate_fn
        )

    print(f"  Test samples     : {len(test_dataset)}\n")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6. Extract training embeddings (no dropout)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    model.eval()
    head.eval()

    print("Extracting training embeddings (no dropout)...")
    train_embeddings_list = []
    train_targets_list = []

    extract_fn = _lvm_extract_batch_embeddings if args.model_type == "lvm" else _siglip_extract_batch_embeddings

    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Train Embeddings"):
            if args.model_type == "lvm":
                batch_convs, batch_labels = batch
            else:
                batch_images, batch_labels = batch

            try:
                if args.model_type == "lvm":
                    preds, vecs = extract_fn(
                        model, processor, batch_convs, head, llm_eos_config, device
                    )
                else:
                    preds, vecs = extract_fn(
                        model, processor, batch_images, head, device
                    )
                train_embeddings_list.append(vecs.float().cpu())
                train_targets_list.append(batch_labels.cpu())
            except RuntimeError:
                print("  WARNING: Skipping batch due to OOM.")
                continue

    train_embeddings_tensor = torch.cat(train_embeddings_list, dim=0)
    train_targets_tensor = torch.cat(train_targets_list, dim=0)
    print(f"  Extracted {train_embeddings_tensor.shape[0]} training embeddings "
          f"with shape {train_embeddings_tensor.shape}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 7. Train KNN and Linear Probe classifiers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\nTraining KNN and Linear Probe classifiers on training embeddings...")

    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(train_embeddings_tensor.numpy(), train_targets_tensor.numpy())
    print(f"  KNN trained with {k_neighbors} neighbors.")

    linear_probe = LogisticRegression(max_iter=linear_probe_max_iter, n_jobs=-1)
    linear_probe.fit(train_embeddings_tensor.numpy(), train_targets_tensor.numpy())
    print(f"  Linear Probe trained (max_iter={linear_probe_max_iter}).")

    # Also build a distance-weighted KNN for probability predictions
    knn_proba = KNeighborsClassifier(n_neighbors=k_neighbors, weights='distance')
    knn_proba.fit(train_embeddings_tensor.numpy(), train_targets_tensor.numpy())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 8. Monte-Carlo Dropout Evaluation on test set
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\nRunning Monte-Carlo Dropout evaluation with {args.mc_samples} samples...")

    # Storage for averaged probabilities
    test_mc_head_probs = np.zeros((len(test_dataset), num_classes), dtype=np.float64)
    test_mc_knn_probs = np.zeros((len(test_dataset), num_classes), dtype=np.float64)
    test_mc_linear_probs = np.zeros((len(test_dataset), num_classes), dtype=np.float64)
    test_labels_array = np.zeros(len(test_dataset), dtype=np.int64)

    # Also keep a single-pass embedding for KNN (dropout on projected embeddings)
    # and for linear probe (trained on non-dropout embeddings)
    
    idx_offset = 0

    model.train()  # Enable dropout in base model's hook layers if applicable
    head.train()   # Enable dropout in head

    for batch in tqdm(test_loader, desc="MC Dropout Eval"):
        if args.model_type == "lvm":
            batch_convs, batch_labels = batch
            batch_size_actual = len(batch_convs)
        else:
            batch_images, batch_labels = batch
            batch_size_actual = len(batch_images)

        test_labels_array[idx_offset:idx_offset + batch_size_actual] = batch_labels.numpy()

        # Step 1: Extract base model features (no dropout in base model)
        model.eval()
        head.eval()
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                if args.model_type == "lvm":
                    from lvm_utils.model_helpers import get_batch_conversation_embeddings_with_config
                    base_features, _ = get_batch_conversation_embeddings_with_config(
                        model=model,
                        processor=processor,
                        conversations=batch_convs,
                        config=llm_eos_config,
                        normalize=True,
                    )
                else:
                    from lvm_utils.model_helpers import get_siglip2_image_embeddings
                    base_features = get_siglip2_image_embeddings(
                        model=model, processor=processor,
                        images=batch_images, normalize=True,
                    )

        # Store base features for non-dropout linear probe evaluation
        # (linear probe was trained on non-dropout embeddings)
        base_features_cpu = base_features.float().cpu()

        # Step 2: Monte-Carlo sampling through the head (with dropout)
        head.train()  # Enable dropout
        mc_head_logits_sum = torch.zeros(batch_size_actual, num_classes, dtype=torch.float64, device=device)
        mc_embeddings_list = []

        for _ in range(args.mc_samples):
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    mc_logits, mc_embeddings = head(base_features)
            mc_head_logits_sum += mc_logits.float()
            mc_embeddings_list.append(mc_embeddings.float().cpu())

        # Average head logits -> probabilities
        avg_head_logits = mc_head_logits_sum / args.mc_samples
        avg_head_probs = F.softmax(avg_head_logits, dim=-1).detach().cpu().numpy()
        test_mc_head_probs[idx_offset:idx_offset + batch_size_actual] = avg_head_probs

        # Average MC embeddings for KNN
        avg_mc_embeddings = torch.stack(mc_embeddings_list, dim=0).mean(dim=0).numpy()
        
        # KNN predictions using averaged MC embeddings
        knn_probs = knn_proba.predict_proba(avg_mc_embeddings)
        test_mc_knn_probs[idx_offset:idx_offset + batch_size_actual] = knn_probs

        # Linear probe: trained on projected embeddings (head output), so evaluate
        # on the MC-averaged projected embeddings for consistency.
        linear_probs = linear_probe.predict_proba(avg_mc_embeddings)
        test_mc_linear_probs[idx_offset:idx_offset + batch_size_actual] = linear_probs

        idx_offset += batch_size_actual

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 9. Calculate and report Top-1 and Top-5 accuracy
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n{'='*60}")
    print(f"Monte-Carlo Dropout Evaluation Results")
    print(f"{'='*60}\n")

    results = {}

    # --- Head (MC Dropout averaged) ---
    head_top1 = _top_k_accuracy(test_mc_head_probs, test_labels_array, k=1)
    head_top5 = _top_k_accuracy(test_mc_head_probs, test_labels_array, k=5)
    print(f"  Linear Head (MC Dropout avg):")
    print(f"    Top-1 Accuracy : {head_top1:.4f}")
    print(f"    Top-5 Accuracy : {head_top5:.4f}")
    results["head_top1"] = head_top1
    results["head_top5"] = head_top5

    # --- KNN (MC Dropout averaged embeddings) ---
    knn_top1 = _top_k_accuracy(test_mc_knn_probs, test_labels_array, k=1)
    knn_top5 = _top_k_accuracy(test_mc_knn_probs, test_labels_array, k=5)
    print(f"\n  KNN (MC Dropout avg embeddings):")
    print(f"    Top-1 Accuracy : {knn_top1:.4f}")
    print(f"    Top-5 Accuracy : {knn_top5:.4f}")
    results["knn_top1"] = knn_top1
    results["knn_top5"] = knn_top5

    # --- Linear Probe (trained on non-dropout embeddings) ---
    lp_top1 = _top_k_accuracy(test_mc_linear_probs, test_labels_array, k=1)
    lp_top5 = _top_k_accuracy(test_mc_linear_probs, test_labels_array, k=5)
    print(f"\n  Linear Probe:")
    print(f"    Top-1 Accuracy : {lp_top1:.4f}")
    print(f"    Top-5 Accuracy : {lp_top5:.4f}")
    results["linear_probe_top1"] = lp_top1
    results["linear_probe_top5"] = lp_top5

    print(f"\n{'='*60}\n")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 10. Save results
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    results_path = os.path.join(args.run_dir, "mc_dropout_eval_results.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "top1_accuracy", "top5_accuracy",
            "mc_samples", "k_neighbors", "num_classes",
            "test_samples", "train_samples",
            "checkpoint_dir", "model_type"
        ])
        writer.writeheader()
        writer.writerow({
            "method": "head_mc_dropout_avg",
            "top1_accuracy": f"{head_top1:.6f}",
            "top5_accuracy": f"{head_top5:.6f}",
            "mc_samples": args.mc_samples,
            "k_neighbors": k_neighbors,
            "num_classes": num_classes,
            "test_samples": len(test_dataset),
            "train_samples": len(train_dataset),
            "checkpoint_dir": ckpt_dir,
            "model_type": args.model_type,
        })
        writer.writerow({
            "method": "knn_mc_dropout_avg_embeddings",
            "top1_accuracy": f"{knn_top1:.6f}",
            "top5_accuracy": f"{knn_top5:.6f}",
            "mc_samples": args.mc_samples,
            "k_neighbors": k_neighbors,
            "num_classes": num_classes,
            "test_samples": len(test_dataset),
            "train_samples": len(train_dataset),
            "checkpoint_dir": ckpt_dir,
            "model_type": args.model_type,
        })
        writer.writerow({
            "method": "linear_probe",
            "top1_accuracy": f"{lp_top1:.6f}",
            "top5_accuracy": f"{lp_top5:.6f}",
            "mc_samples": args.mc_samples,
            "k_neighbors": k_neighbors,
            "num_classes": num_classes,
            "test_samples": len(test_dataset),
            "train_samples": len(train_dataset),
            "checkpoint_dir": ckpt_dir,
            "model_type": args.model_type,
        })

    print(f"Results saved to: {results_path}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()