import json
import csv
from pathlib import Path
from collections import Counter
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = BASE_DIR / "data" / "images"
META_DIR = BASE_DIR / "data" / "metadata"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze():
    image_files = {
        p.stem: p
        for p in IMAGE_DIR.glob("*.jpg")
    }

    metadata_files = {
        p.stem: p
        for p in META_DIR.glob("*.json")
    }

    image_ids = set(image_files)
    metadata_ids = set(metadata_files)

    matched_ids = sorted(image_ids & metadata_ids)
    missing_images = sorted(metadata_ids - image_ids)
    missing_metadata = sorted(image_ids - metadata_ids)

    rows = []
    expected_quantities = []
    total_inventory_quantities = []

    corrupted_images = []

    for image_id in matched_ids:
        image_path = image_files[image_id]
        metadata_path = metadata_files[image_id]

        # Check image
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception:
            corrupted_images.append(image_id)

        # Read metadata
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            expected_quantity = data.get("EXPECTED_QUANTITY")

            inventory = data.get("BIN_FCSKU_DATA", {})

            total_quantity = sum(
                item.get("quantity", 0)
                for item in inventory.values()
                if isinstance(item, dict)
            )

            product_names = [
                item.get("name", "")
                for item in inventory.values()
                if isinstance(item, dict)
            ]

            asins = list(inventory.keys())

            expected_quantities.append(expected_quantity)
            total_inventory_quantities.append(total_quantity)

            rows.append({
                "image_id": image_id,
                "image_path": str(image_path),
                "metadata_path": str(metadata_path),
                "expected_quantity": expected_quantity,
                "total_inventory_quantity": total_quantity,
                "num_unique_products": len(inventory),
                "product_asins": "|".join(asins),
                "product_names": "|".join(product_names),
            })

        except Exception as e:
            print(f"Metadata error: {metadata_path}: {e}")

    # Save dataset CSV
    dataset_csv = PROCESSED_DIR / "dataset.csv"

    fieldnames = [
        "image_id",
        "image_path",
        "metadata_path",
        "expected_quantity",
        "total_inventory_quantity",
        "num_unique_products",
        "product_asins",
        "product_names",
    ]

    with open(dataset_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    summary = {
        "total_images": len(image_files),
        "total_metadata": len(metadata_files),
        "matched_pairs": len(matched_ids),
        "missing_images": len(missing_images),
        "missing_metadata": len(missing_metadata),
        "corrupted_images": len(corrupted_images),
        "expected_quantity_min": min(expected_quantities)
        if expected_quantities else None,
        "expected_quantity_max": max(expected_quantities)
        if expected_quantities else None,
        "expected_quantity_mean": round(
            sum(expected_quantities) / len(expected_quantities), 2
        )
        if expected_quantities else None,
        "expected_quantity_distribution":
            dict(sorted(Counter(expected_quantities).items())),
        "total_inventory_quantity_min":
            min(total_inventory_quantities)
            if total_inventory_quantities else None,
        "total_inventory_quantity_max":
            max(total_inventory_quantities)
            if total_inventory_quantities else None,
        "total_inventory_quantity_mean":
            round(
                sum(total_inventory_quantities)
                / len(total_inventory_quantities),
                2
            )
            if total_inventory_quantities else None,
    }

    summary_path = OUTPUT_DIR / "dataset_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\n===== DATASET ANALYSIS =====")
    print(f"Images              : {len(image_files)}")
    print(f"Metadata            : {len(metadata_files)}")
    print(f"Matched pairs       : {len(matched_ids)}")
    print(f"Missing images      : {len(missing_images)}")
    print(f"Missing metadata    : {len(missing_metadata)}")
    print(f"Corrupted images    : {len(corrupted_images)}")

    print("\n===== EXPECTED QUANTITY =====")
    print(f"Minimum             : {summary['expected_quantity_min']}")
    print(f"Maximum             : {summary['expected_quantity_max']}")
    print(f"Mean                : {summary['expected_quantity_mean']}")

    print("\nDistribution:")
    for quantity, count in sorted(
        Counter(expected_quantities).items()
    ):
        print(f"  {quantity}: {count}")

    print("\n===== INVENTORY QUANTITY =====")
    print(f"Minimum             : {summary['total_inventory_quantity_min']}")
    print(f"Maximum             : {summary['total_inventory_quantity_max']}")
    print(f"Mean                : {summary['total_inventory_quantity_mean']}")

    print("\nCreated:")
    print(dataset_csv)
    print(summary_path)


if __name__ == "__main__":
    analyze()