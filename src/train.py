"""
Amazon Bin Inventory AI - Model Training Pipeline
Trains ResNet-18 on bin inventory images with validation tracking and best checkpoint saving.
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


from src.config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    MODELS_DIR,
    NUM_CLASSES,
    WEIGHT_DECAY,
)
from src.dataset import get_dataloaders, prepare_dataset_splits
from src.model import build_model
from src.utils import plot_training_history


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs a single training epoch across batches."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="  Training", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluates model performance on the validation split."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = correct / total if total > 0 else 0.0
    return val_loss, val_acc


def run_training(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    checkpoint_path: Path = BEST_MODEL_PATH,
    device: torch.device = DEVICE,
) -> Dict[str, List[float]]:
    """
    Executes complete training pipeline:
    1. Loads dataset splits
    2. Initializes ResNet-18 model
    3. Runs training & validation loops
    4. Saves optimal model weights
    5. Produces training curves plot
    """
    print(f"🚀 Starting Training on Device: {device}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure splits are present
    try:
        train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)
    except FileNotFoundError:
        print("📂 Splitting dataset into train/val/test...")
        prepare_dataset_splits()
        train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    # Initialize model
    model = build_model(num_classes=NUM_CLASSES, pretrained=True, device=device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")

    print(f"📊 Training parameters: Epochs={epochs}, Batch Size={batch_size}, LR={learning_rate}")
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        )

        # Save checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                checkpoint_path,
            )
            print(f"  ⭐ New best model saved to: {checkpoint_path}")

    # Generate training curves
    plot_training_history(history)
    print("✅ Training complete!")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Amazon Bin Inventory AI model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")

    args = parser.parse_args()
    run_training(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
