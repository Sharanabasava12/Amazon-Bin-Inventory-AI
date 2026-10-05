"""
Amazon Bin Inventory AI - Utilities & Visualization
Helper functions for training metrics, confusion matrix plotting, and image overlays.
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config import CLASS_NAMES, OUTPUTS_DIR



def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
) -> Path:
    """
    Plots training and validation loss & accuracy curves across epochs.
    """
    if save_path is None:
        save_path = OUTPUTS_DIR / "training_curves.png"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#e74c3c", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#3498db", linewidth=2)
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # Accuracy plot
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Acc", color="#2ecc71", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s--", label="Val Acc", color="#9b59b6", linewidth=2)
    ax2.set_title("Top-1 Accuracy (%) vs. Epochs", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"📈 Saved training curves plot to {save_path}")
    return save_path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    save_path: Optional[Path] = None,
    normalize: bool = False,
) -> Path:
    """
    Renders and saves a confusion matrix heatmap.
    """
    if save_path is None:
        save_path = OUTPUTS_DIR / "confusion_matrix.png"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
        fmt = ".2f"
        title = "Normalized Confusion Matrix (Bin Item Count)"
    else:
        fmt = "d"
        title = "Confusion Matrix (Bin Item Count)"

    fig, ax = plt.subplots(figsize=(8, 7))

    try:
        import seaborn as sns
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            ax=ax,
        )
    except Exception:
        # Fallback pure Matplotlib heatmap
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = f"{cm[i, j]:{fmt}}"
                ax.text(j, i, val, ha="center", va="center", color="black")

        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Count Class", fontsize=11, labelpad=8)
    ax.set_ylabel("True Ground Truth Class", fontsize=11, labelpad=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"📊 Saved confusion matrix plot to {save_path}")
    return save_path


def annotate_bin_image(
    image: Image.Image,
    predicted_count: int,
    expected_qty: int,
    status: str,
) -> Image.Image:
    """
    Draws a visual audit banner onto the image for clear warehouse operator display.
    """
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Color palette based on status
    if status == "MATCHED":
        banner_color = (46, 204, 113)  # Emerald green
        badge_text = "VERIFIED MATCH"
    elif status == "SHORTAGE":
        banner_color = (231, 76, 60)   # Red
        badge_text = "SHORTAGE ALERT"
    else:
        banner_color = (241, 196, 15)  # Amber
        badge_text = "OVERAGE ALERT"

    # Draw top banner
    banner_height = max(40, int(h * 0.08))
    draw.rectangle([0, 0, w, banner_height], fill=banner_color)

    # Info text
    info_text = f"[{badge_text}] Pred: {predicted_count} | Recorded: {expected_qty}"
    draw.text((15, int(banner_height * 0.25)), info_text, fill=(255, 255, 255))

    return img
