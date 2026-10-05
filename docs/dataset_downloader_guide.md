# 📥 Amazon Bin Dataset Downloader Guide (`src/download_data.py`)

This document explains the architecture, design decisions, and usage of the automated dataset downloader for the **Amazon Bin Image Dataset** (`aft-vbi-pds`).

---

## 1. Dataset Overview

The **Amazon Fulfillment Technologies Visual Bin Inspection Public Dataset** (`aft-vbi-pds`) is an open dataset hosted on AWS S3:
- **S3 URI**: `s3://aft-vbi-pds/`
- **HTTP Base URL**: `https://aft-vbi-pds.s3.amazonaws.com/`
- **Total Images**: Over 500,000 tote bin camera captures (>100 GB).

Each sample in the dataset is a pair consisting of:
1. **An Image File**: Located at `bin-images/<id>.jpg` (e.g., `00001.jpg`).
2. **A Metadata File**: Located at `metadata/<id>.json` (e.g., `00001.json`).

### JSON Metadata Structure
```json
{
  "EXPECTED_QUANTITY": 12,
  "BIN_FCSKU_DATA": {
    "B000C33MI2": {
      "asin": "B000C33MI2",
      "name": "FRAM XG7317 ULTRA Spin-On Oil Filter with Sure Grip",
      "quantity": 4,
      "weight": { "unit": "pounds", "value": 0.3 },
      "dimensions": { "length": 3.9, "width": 2.9, "height": 2.8 }
    }
  }
}
```

---

## 2. How the Downloader Works

The downloader script ([src/download_data.py](file:///f:/5th%20sem/Machine%20Learning/Amazon-Bin-Inventory-AI/src/download_data.py)) is engineered to be **safe, beginner-friendly, and network-efficient**.

### Key Architectural Features:

```
[Request Sample ID (e.g. 00001)]
           │
           ▼
[Check Local Files Exist?]
   ├── YES ──► Skip download (Instant Cache Hit)
   │
   └── NO  ──► 1. Download metadata/00001.json.tmp
               2. Validate JSON & check EXPECTED_QUANTITY
               3. Atomic rename to metadata/00001.json
               4. Download bin-images/00001.jpg.tmp
               5. Atomic rename to bin-images/00001.jpg
               6. Update tqdm progress bar
```

1. **Direct HTTP Access (No AWS Keys Needed)**:
   The dataset is in an open S3 bucket. We download directly via standard HTTPS without requiring AWS credentials, `awscli`, or `boto3`.

2. **5-Digit Zero-Padding**:
   The S3 keys in `aft-vbi-pds` use 5-digit formatted IDs (e.g., ID `1` becomes `00001.jpg` and `00001.json`). The helper function `get_sample_filenames()` automatically handles this format.

3. **Atomic File Writing**:
   Files are initially written to `.tmp` files and only renamed to `.jpg` or `.json` upon successful completion. This guarantees that interrupted downloads or network drops will never leave corrupted, half-downloaded files on your disk.

4. **Cache & Skip Logic**:
   If `data/images/00001.jpg` and `data/metadata/00001.json` already exist on your disk, the script skips them immediately. You can safely stop and resume the downloader at any time!

5. **JSON Validation & Orphan Cleanup**:
   The script downloads the metadata file first and verifies that the JSON contains `EXPECTED_QUANTITY`. If an image fails to download, any orphaned metadata is safely removed.

6. **Interactive Progress Bar**:
   Uses `tqdm` to display live download speeds, percentage completion, cached count, and the active sample ID.

---

## 3. Command-Line Usage

### 3.1 Default Usage (Downloads 2000 pairs)
```bash
python src/download_data.py --samples 2000
```
This stores images in `data/images/` and JSON files in `data/metadata/`.

### 3.2 Quick Test (Small batch of 10 samples)
```bash
python src/download_data.py --samples 10
```

### 3.3 Custom Range & Starting ID
If you want to download a different shard or start from sample ID 500:
```bash
python src/download_data.py --samples 500 --start_id 500
```

### 3.4 Custom Output Directories
```bash
python src/download_data.py --samples 100 --images_dir data/my_images --metadata_dir data/my_metadata
```

---

## 4. Summary of Script Arguments

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--samples` | `int` | `2000` | Number of complete image/metadata pairs to download |
| `--start_id` | `int` | `1` | Starting sample numerical ID |
| `--images_dir` | `str` | `data/images` | Directory where `.jpg` files are saved |
| `--metadata_dir` | `str` | `data/metadata` | Directory where `.json` files are saved |

---

## 5. Storage Recommendation
- 10 samples $\approx$ 500 KB
- 2,000 samples $\approx$ 100 MB – 150 MB (ideal for fast local training on personal laptops/desktops)
- 5,000 samples $\approx$ 300 MB – 400 MB
