"""
Amazon Bin Inventory AI - Configuration Module
Defines all directory paths, hyperparameter defaults, and dataset constants.
"""

from pathlib import Path
import sys
import torch

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Base directories

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
METADATA_DIR = DATA_DIR / "metadata"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


# Create directories if they do not exist
for directory in [
    RAW_DIR,
    IMAGES_DIR,
    METADATA_DIR,
    PROCESSED_DIR,
    SAMPLES_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset & Class definitions
NUM_CLASSES = 6
CLASS_NAMES = ["0 items", "1 item", "2 items", "3 items", "4 items", "5+ items"]
MAX_COUNT_CAP = 5  # Counts >= 5 are mapped to class index 5 (5+ items)

# Model & Training parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 10
NUM_WORKERS = 0  # 0 is safest for Windows multiprocessing

# Normalization constants (ImageNet defaults)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default best model checkpoint path
BEST_MODEL_PATH = MODELS_DIR / "bin_resnet18_best.pth"

# AWS Public S3 Dataset Configuration (aft-vbi-pds)
S3_BASE_URL = "https://aft-vbi-pds.s3.amazonaws.com"
S3_IMAGE_URL = f"{S3_BASE_URL}/bin-images"
S3_METADATA_URL = f"{S3_BASE_URL}/metadata"

# Compute device detection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_class_index(count: int) -> int:
    """Maps integer item count to class index (0 to 5)."""
    return min(max(0, int(count)), MAX_COUNT_CAP)


def get_class_label(class_idx: int) -> str:
    """Returns friendly label name for class index."""
    if 0 <= class_idx < len(CLASS_NAMES):
        return CLASS_NAMES[class_idx]
    return f"{class_idx} items"
