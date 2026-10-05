"""
Amazon Bin Inventory AI - Dataset Downloader (aft-vbi-pds)
==========================================================
Downloads paired bin images and JSON metadata from the public Amazon
Fulfillment Technologies Visual Bin Inspection dataset (s3://aft-vbi-pds).

Features:
- Configurable sample count (default: 2000 pairs)
- Direct HTTP download from public S3 (no AWS credentials required)
- Stores images in data/images/ and metadata in data/metadata/
- Skips already downloaded files to save time and bandwidth
- Atomic writing to avoid incomplete or corrupted files
- Progress tracking with tqdm
- Safe handling of missing keys or network timeouts
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Optional, Tuple
import requests
from tqdm import tqdm

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# S3 Public HTTP Endpoints
S3_BASE_URL = "https://aft-vbi-pds.s3.amazonaws.com"
S3_IMAGES_URL = f"{S3_BASE_URL}/bin-images"
S3_METADATA_URL = f"{S3_BASE_URL}/metadata"

# Default local storage paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "images"
DEFAULT_METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


def download_file(url: str, destination_path: Path, timeout: int = 15) -> bool:
    """
    Downloads a single file via HTTP with streaming.
    Uses a temporary file to guarantee atomic writes (no half-downloaded files).

    Args:
        url: Direct HTTP URL to the file on S3
        destination_path: Local Path where the file should be saved
        timeout: Request timeout in seconds

    Returns:
        bool: True if downloaded successfully, False otherwise
    """
    if destination_path.exists():
        return True  # Already downloaded

    temp_path = destination_path.with_suffix(destination_path.suffix + ".tmp")

    try:
        response = requests.get(url, stream=True, timeout=timeout)
        if response.status_code != 200:
            return False

        # Stream write in chunks of 8KB
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Atomic rename after complete download
        temp_path.replace(destination_path)
        return True

    except Exception:
        # Clean up temporary file on failure
        if temp_path.exists():
            temp_path.unlink()
        return False


def get_sample_filenames(sample_id: int) -> Tuple[str, str]:
    """
    Generates standard Amazon Bin dataset filenames.
    Files in aft-vbi-pds are 5-digit zero-padded (e.g., 00001.jpg, 00001.json).
    """
    stem = f"{sample_id:05d}"
    return f"{stem}.jpg", f"{stem}.json"


def download_single_pair(
    sample_id: int,
    images_dir: Path,
    metadata_dir: Path,
    timeout: int = 15,
) -> Tuple[bool, bool]:
    """
    Downloads a single image and its corresponding JSON metadata file.

    Args:
        sample_id: Integer identifier (e.g., 1 -> 00001)
        images_dir: Directory to save image
        metadata_dir: Directory to save JSON
        timeout: Network timeout

    Returns:
        (success: bool, already_existed: bool)
    """
    img_name, meta_name = get_sample_filenames(sample_id)
    img_path = images_dir / img_name
    meta_path = metadata_dir / meta_name

    # Check if both already exist locally
    if img_path.exists() and meta_path.exists():
        return True, True

    meta_url = f"{S3_METADATA_URL}/{meta_name}"
    img_url = f"{S3_IMAGES_URL}/{img_name}"

    # 1. Download metadata first
    meta_downloaded = download_file(meta_url, meta_path, timeout=timeout)
    if not meta_downloaded:
        return False, False

    # Validate JSON syntax
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "EXPECTED_QUANTITY" not in data:
            meta_path.unlink(missing_ok=True)
            return False, False
    except Exception:
        meta_path.unlink(missing_ok=True)
        return False, False

    # 2. Download corresponding image
    img_downloaded = download_file(img_url, img_path, timeout=timeout)
    if not img_downloaded:
        # If image fails, remove orphaned metadata
        meta_path.unlink(missing_ok=True)
        return False, False

    return True, False


def download_amazon_bin_data(
    num_samples: int = 2000,
    start_id: int = 1,
    images_dir: Path = DEFAULT_IMAGES_DIR,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
) -> int:
    """
    Main download loop that fetches 'num_samples' image/metadata pairs.

    Args:
        num_samples: Target count of complete image + JSON pairs to download
        start_id: Starting ID number (default: 1)
        images_dir: Destination folder for images
        metadata_dir: Destination folder for JSON files

    Returns:
        int: Total number of valid pairs available on disk
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("📦 Amazon Bin Image Dataset Downloader (aft-vbi-pds)")
    print("=" * 65)
    print(f"Target Samples:   {num_samples}")
    print(f"Starting ID:      {start_id}")
    print(f"Images Target:    {images_dir.resolve()}")
    print(f"Metadata Target:  {metadata_dir.resolve()}")
    print(f"Source S3 Bucket: s3://aft-vbi-pds/")
    print("-" * 65)

    downloaded_count = 0
    skipped_count = 0
    missing_count = 0
    current_id = start_id

    # Maximum IDs to scan before giving up (safety stop)
    max_scan_limit = start_id + (num_samples * 5)

    with tqdm(total=num_samples, desc="Downloading Bin Pairs", unit="pair") as pbar:
        while downloaded_count + skipped_count < num_samples and current_id <= max_scan_limit:
            success, already_existed = download_single_pair(
                sample_id=current_id,
                images_dir=images_dir,
                metadata_dir=metadata_dir,
            )

            if success:
                if already_existed:
                    skipped_count += 1
                else:
                    downloaded_count += 1
                pbar.update(1)
                pbar.set_postfix({
                    "new": downloaded_count,
                    "cached": skipped_count,
                    "id": f"{current_id:05d}",
                })
            else:
                missing_count += 1

            current_id += 1

    total_ready = downloaded_count + skipped_count

    print("\n" + "=" * 65)
    print("✅ DOWNLOAD SUMMARY")
    print("=" * 65)
    print(f"New Pairs Downloaded:   {downloaded_count}")
    print(f"Existing Pairs Kept:    {skipped_count}")
    print(f"Total Available Pairs:  {total_ready}/{num_samples}")
    print(f"Missing/Skipped IDs:    {missing_count}")
    print(f"Saved Images In:        {images_dir}")
    print(f"Saved Metadata In:      {metadata_dir}")
    print("=" * 65)

    return total_ready


def main():
    parser = argparse.ArgumentParser(
        description="Download image/metadata pairs from public Amazon Bin Dataset (aft-vbi-pds)."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of image/metadata pairs to download (default: 2000)",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=1,
        help="Starting sample ID number (default: 1)",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=str(DEFAULT_IMAGES_DIR),
        help=f"Directory to save images (default: {DEFAULT_IMAGES_DIR})",
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default=str(DEFAULT_METADATA_DIR),
        help=f"Directory to save JSON metadata (default: {DEFAULT_METADATA_DIR})",
    )

    args = parser.parse_args()

    download_amazon_bin_data(
        num_samples=args.samples,
        start_id=args.start_id,
        images_dir=Path(args.images_dir),
        metadata_dir=Path(args.metadata_dir),
    )


if __name__ == "__main__":
    main()
