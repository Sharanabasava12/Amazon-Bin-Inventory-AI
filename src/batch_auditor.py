from pathlib import Path
import json
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn


CLASSES = [0, 1, 2, 3, 4, 5]

IMAGE_SIZE = 224


def load_model(project_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    model_path = project_root / "models" / "resnet34_best.pth"

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, device


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict_image(model, device, image_path, transform):
    try:
        image = Image.open(image_path).convert("RGB")

        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)

            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )

        predicted_class = CLASSES[predicted.item()]

        return predicted_class, float(confidence.item())

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, 0.0


def read_expected_quantity(metadata_path):
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return int(data.get("EXPECTED_QUANTITY", 0))

    except Exception:
        return None


def get_status(predicted, expected):
    if predicted is None or expected is None:
        return "ERROR"

    # 5 represents the model's "5+" class.
    predicted_numeric = predicted

    difference = predicted_numeric - expected

    if difference == 0:
        return "MATCHED"
    elif difference < 0:
        return "SHORTAGE"
    else:
        return "OVERAGE"


def run_batch_audit(project_root, max_images=None):

    image_dir = project_root / "data" / "images"
    metadata_dir = project_root / "data" / "metadata"
    output_dir = project_root / "outputs"

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(image_dir.glob("*.jpg"))

    if max_images is not None:
        image_files = image_files[:max_images]

    print("=" * 50)
    print("BATCH INVENTORY AUDITOR")
    print("=" * 50)

    print(f"Images found: {len(image_files)}")

    model, device = load_model(project_root)

    print(f"Device: {device}")

    transform = get_transform()

    results = []

    for index, image_path in enumerate(image_files, start=1):

        bin_id = image_path.stem

        metadata_path = metadata_dir / f"{bin_id}.json"

        expected = read_expected_quantity(metadata_path)

        predicted, confidence = predict_image(
            model,
            device,
            image_path,
            transform
        )

        status = get_status(predicted, expected)

        difference = None

        if predicted is not None and expected is not None:
            difference = predicted - expected

        results.append({
            "bin_id": bin_id,
            "image": image_path.name,
            "expected_quantity": expected,
            "predicted_quantity": predicted,
            "confidence": round(confidence, 4),
            "difference": difference,
            "status": status
        })

        if index % 25 == 0 or index == len(image_files):
            print(f"Processed {index}/{len(image_files)}")

    df = pd.DataFrame(results)

    csv_path = output_dir / "batch_audit_results.csv"

    df.to_csv(csv_path, index=False)

    print()
    print("=" * 50)
    print("BATCH AUDIT COMPLETE")
    print("=" * 50)

    print(f"Total bins     : {len(df)}")
    print(f"Matched        : {(df['status'] == 'MATCHED').sum()}")
    print(f"Shortages      : {(df['status'] == 'SHORTAGE').sum()}")
    print(f"Overages       : {(df['status'] == 'OVERAGE').sum()}")
    print(f"Errors         : {(df['status'] == 'ERROR').sum()}")

    print()
    print(f"Saved to:")
    print(csv_path)

    return df


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent

    run_batch_audit(project_root)