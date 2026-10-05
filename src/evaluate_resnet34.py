import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import create_dataloaders
from model import create_model


def main():

    base_dir = Path(__file__).resolve().parent.parent

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Load test data
    _, _, test_loader = create_dataloaders(
        base_dir,
        batch_size=16
    )

    # Create model
    model = create_model(pretrained=False)

    checkpoint = (
        base_dir / "models" / "resnet34_best.pth"
    )

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            ).cpu().numpy()

            y_pred.extend(predictions)
            y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Accuracy
    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    # RMSE
    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4, 5]
    )

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4, 5],
        zero_division=0
    )

    print("\n==============================")
    print("RESNET-34 TEST RESULTS")
    print("==============================")

    print(
        f"Accuracy : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"RMSE     : {rmse:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    # Save results
    output_dir = base_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    results = {
        "model": "ResNet-34",
        "test_samples": len(y_true),
        "accuracy": float(accuracy),
        "rmse": float(rmse),
        "confusion_matrix": cm.tolist(),
        "true_labels": y_true.tolist(),
        "predicted_labels": y_pred.tolist()
    }

    output_file = (
        output_dir /
        "resnet34_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        f"\nResults saved to:\n{output_file}"
    )


if __name__ == "__main__":
    main()