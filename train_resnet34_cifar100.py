"""
Train and evaluate ResNet34 on CIFAR-100 with 6,000 training samples (stratified).

- Stratified subset: 60 samples per class
- Pretrained ResNet34 weights
- NAdam optimizer, 10 epochs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet34, ResNet34_Weights
import numpy as np
from collections import defaultdict
import random


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_stratified_indices(dataset, samples_per_class=60):
    """
    Return indices forming a stratified subset with exactly `samples_per_class`
    samples per class.
    """
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_to_indices[label].append(idx)

    selected = []
    for cls in sorted(class_to_indices.keys()):
        indices = class_to_indices[cls]
        if len(indices) < samples_per_class:
            raise ValueError(
                f"Class {cls} has only {len(indices)} samples, "
                f"cannot select {samples_per_class}."
            )
        selected.extend(random.sample(indices, samples_per_class))

    return selected


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Transforms ---
    # ResNet expects 224x224 images; CIFAR-100 is 32x32, so we resize.
    # Use ImageNet statistics as the pretrained weights expect them.
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # --- Datasets ---
    train_dataset_full = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=test_transform
    )

    # Stratified subset: 60 samples per class = 6000 total
    print("Creating stratified training subset (60 samples per class)...")
    train_indices = get_stratified_indices(
        datasets.CIFAR100(root="./data", train=True, download=True),
        samples_per_class=60,
    )
    train_dataset = Subset(train_dataset_full, train_indices)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples:     {len(test_dataset)}")

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )

    # --- Model ---
    print("Loading pretrained ResNet34...")
    model = resnet34()
    # Replace the final FC layer for CIFAR-100 (100 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 100)
    model = model.to(device)

    # --- Loss, Optimizer, Scheduler ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.NAdam(model.parameters(), lr=1e-4)

    # --- Training ---
    num_epochs = 10
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        if test_acc > best_acc:
            best_acc = test_acc

        print(
            f"Epoch {epoch:2d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

    print(f"\nFinal Best Test Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()