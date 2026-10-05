"""
Amazon Bin Inventory AI - Dataset Downloader & Sample Generator
Downloads genuine samples from the public AWS S3 bucket (aft-vbi-pds)
and provides synthetic sample generation for testing without internet dependencies.
"""

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Tuple

# Add repository root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont
import requests

from src.config import (
    IMAGES_DIR,
    METADATA_DIR,
    RAW_DIR,
    S3_IMAGE_URL,
    S3_METADATA_URL,
    SAMPLES_DIR,
    get_class_index,
)



def download_single_sample(
    sample_id: int,
    image_dir: Path = IMAGES_DIR,
    metadata_dir: Path = METADATA_DIR,
    timeout: int = 10,
) -> bool:
    """
    Downloads an image and its corresponding JSON metadata from public AWS S3.
    """
    img_filename = f"{sample_id}.jpg"
    meta_filename = f"{sample_id}.json"

    img_url = f"{S3_IMAGE_URL}/{img_filename}"
    meta_url = f"{S3_METADATA_URL}/{meta_filename}"

    img_path = image_dir / img_filename
    meta_path = metadata_dir / meta_filename

    # Skip if already exists
    if img_path.exists() and meta_path.exists():
        return True

    try:
        # Download metadata first
        meta_resp = requests.get(meta_url, timeout=timeout)
        if meta_resp.status_code != 200:
            return False

        # Validate JSON content
        meta_data = meta_resp.json()
        if "EXPECTED_QUANTITY" not in meta_data:
            return False

        # Download image
        img_resp = requests.get(img_url, timeout=timeout)
        if img_resp.status_code != 200:
            return False

        # Save files
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(meta_resp.text)

        with open(img_path, "wb") as f:
            f.write(img_resp.content)

        return True

    except Exception:
        # Clean up partial files if failed
        if meta_path.exists():
            meta_path.unlink()
        if img_path.exists():
            img_path.unlink()
        return False


def download_dataset_subset(
    num_samples: int = 200,
    start_id: int = 1,
    image_dir: Path = IMAGES_DIR,
    metadata_dir: Path = METADATA_DIR,
) -> int:
    """
    Downloads a specified number of samples from the aft-vbi-pds dataset.
    """
    print(f"📥 Starting download of up to {num_samples} samples from AWS S3...")
    image_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    current_id = start_id
    max_scan = start_id + num_samples * 4  # Scan window in case some IDs are missing

    while downloaded < num_samples and current_id <= max_scan:
        success = download_single_sample(
            current_id, image_dir=image_dir, metadata_dir=metadata_dir
        )
        if success:
            downloaded += 1
            if downloaded % 25 == 0 or downloaded == num_samples:
                print(f"  ✓ Downloaded {downloaded}/{num_samples} samples (current ID: {current_id})")
        current_id += 1

    print(f"✅ Download complete! Saved {downloaded} samples to {RAW_DIR}")
    return downloaded


# Predefined mock product database for synthetic generation & realistic simulation
MOCK_PRODUCTS = [
    {"name": "Anker USB-C Fast Charger 20W", "asin": "B082V32Q8W", "color": (40, 120, 220), "category": "Electronics", "size": [8, 5, 4]},
    {"name": "Stanley Classic Vacuum Insulated Thermos", "asin": "B000FZX944", "color": (60, 100, 70), "category": "Home & Kitchen", "size": [28, 9, 9]},
    {"name": "Logitech MX Master 3S Wireless Mouse", "asin": "B09HM94VDS", "color": (50, 50, 50), "category": "Office", "size": [12, 8, 5]},
    {"name": "Casio Vintage Digital Watch Gold", "asin": "B000GAYQKY", "color": (210, 180, 50), "category": "Watches", "size": [10, 7, 6]},
    {"name": "Hydro Flask Wide Mouth Water Bottle 32oz", "asin": "B01KXHG3S2", "color": (200, 70, 70), "category": "Sports", "size": [24, 9, 9]},
    {"name": "Moleskine Classic Hard Cover Notebook", "asin": "B002AP56A8", "color": (30, 30, 30), "category": "Stationery", "size": [21, 13, 2]},
    {"name": "Bose QuietComfort Noise Cancelling Earbuds", "asin": "B098FH5P3C", "color": (230, 230, 230), "category": "Audio", "size": [7, 6, 3]},
    {"name": "CeraVe Daily Moisturizing Lotion 12oz", "asin": "B000YJ2SLG", "color": (240, 245, 255), "category": "Beauty", "size": [18, 7, 4]},
]


def generate_synthetic_bin(
    sample_id: int,
    visual_item_count: int,
    expected_quantity: int,
    image_dir: Path,
    metadata_dir: Path,
    img_size: Tuple[int, int] = (600, 600),
) -> None:
    """
    Generates a realistic synthetic bin image and matching Amazon metadata JSON.
    Used for instant offline demonstration and testing mismatch detection.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create Bin Background (Amazon Yellow / Dark Gray Tote interior)
    img = Image.new("RGB", img_size, color=(210, 175, 45))  # Amazon Yellow Tote
    draw = ImageDraw.Draw(img)

    # Inner tote shadow / depth perspective
    margin = 50
    draw.rectangle(
        [margin, margin, img_size[0] - margin, img_size[1] - margin],
        fill=(55, 55, 60),  # Dark bin bottom
        outline=(170, 140, 30),
        width=8,
    )

    # Bin texture / grid ribs
    for y in range(margin + 40, img_size[1] - margin, 60):
        draw.line([(margin, y), (img_size[0] - margin, y)], fill=(45, 45, 50), width=2)
    for x in range(margin + 40, img_size[0] - margin, 60):
        draw.line([(x, margin), (x, img_size[1] - margin)], fill=(45, 45, 50), width=2)

    # 2. Draw Visual Items inside the tote
    chosen_products = random.sample(MOCK_PRODUCTS, min(visual_item_count, len(MOCK_PRODUCTS)))
    sku_entries = {}

    placed_boxes = []
    for i in range(visual_item_count):
        prod = chosen_products[i % len(chosen_products)]
        w = random.randint(80, 160)
        h = random.randint(80, 160)

        # Random position inside the inner bin
        for _ in range(20):
            x1 = random.randint(margin + 20, img_size[0] - margin - w - 20)
            y1 = random.randint(margin + 20, img_size[1] - margin - h - 20)
            x2 = x1 + w
            y2 = y1 + h

            # Check overlap tolerance
            overlap = False
            for bx1, by1, bx2, by2 in placed_boxes:
                if not (x2 < bx1 or x1 > bx2 or y2 < by1 or y1 > by2):
                    overlap = True
                    break
            if not overlap or len(placed_boxes) >= 4:
                break

        placed_boxes.append((x1, y1, x2, y2))

        # Draw product box
        item_color = prod["color"]
        draw.rectangle([x1, y1, x2, y2], fill=item_color, outline=(20, 20, 20), width=3)
        # Barcode / label graphic on package
        draw.rectangle([x1 + 10, y1 + 10, x1 + 45, y1 + 30], fill=(245, 245, 245))
        for line_x in range(x1 + 14, x1 + 42, 4):
            draw.line([(line_x, y1 + 12), (line_x, y1 + 28)], fill=(10, 10, 10), width=2)

        # SKU record
        sku_key = prod["asin"]
        sku_entries[sku_key] = {
            "name": prod["name"],
            "asin": prod["asin"],
            "product_type": prod["category"],
            "dimensions": {"width": prod["size"][0], "length": prod["size"][1], "height": prod["size"][2]},
        }

    # 3. Create JSON Metadata matching aft-vbi-pds schema
    metadata = {
        "BIN_FCSKU_DATA": sku_entries,
        "EXPECTED_QUANTITY": int(expected_quantity),
        "bin_info": {
            "bin_id": f"BIN_{sample_id:05d}",
            "dimensions": {"height": 30.5, "length": 45.7, "width": 38.1},
        },
        "_visual_count": int(visual_item_count),  # Ground truth visual label
    }

    # Save image and json
    img.save(image_dir / f"{sample_id}.jpg", quality=90)
    with open(metadata_dir / f"{sample_id}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def create_sample_dataset(
    num_samples: int = 120,
    target_dir: Path = RAW_DIR,
    include_discrepancies: bool = True,
) -> None:
    """
    Generates a full synthetic demo dataset with a balanced distribution
    across item counts 0 to 5+, including deliberate inventory mismatches.
    """
    img_dir = target_dir / "images"
    meta_dir = target_dir / "metadata"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Generating {num_samples} balanced bin samples...")
    for i in range(1, num_samples + 1):
        # Balanced count distribution: 0 to 5 items
        count = (i - 1) % 6

        # Decide whether to create an inventory mismatch (e.g. 20% of samples)
        expected = count
        if include_discrepancies and (i % 5 == 0):
            # 50% shortage, 50% overage
            if count > 0 and random.random() < 0.5:
                expected = count + 1  # Shortage (missing 1)
            else:
                expected = max(0, count - 1)  # Overage (extra 1)

        generate_synthetic_bin(
            sample_id=i,
            visual_item_count=count,
            expected_quantity=expected,
            image_dir=img_dir,
            metadata_dir=meta_dir,
        )

    print(f"✅ Successfully created {num_samples} samples in {target_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or generate Amazon Bin Dataset samples")
    parser.add_argument("--mode", choices=["download", "generate", "sample_suite"], default="sample_suite",
                        help="Operation mode: 'download' from S3, 'generate' local dataset, or 'sample_suite'")
    parser.add_argument("--num_samples", type=int, default=150, help="Number of samples to process")
    parser.add_argument("--start_id", type=int, default=1, help="Starting sample ID for S3 download")

    args = parser.parse_args()

    if args.mode == "download":
        download_dataset_subset(num_samples=args.num_samples, start_id=args.start_id)
    elif args.mode == "generate":
        create_sample_dataset(num_samples=args.num_samples)
    elif args.mode == "sample_suite":
        # Create quick samples in data/samples for Streamlit demo
        create_sample_dataset(num_samples=24, target_dir=SAMPLES_DIR)
        # Create dataset in data/raw for training
        create_sample_dataset(num_samples=args.num_samples, target_dir=RAW_DIR)
