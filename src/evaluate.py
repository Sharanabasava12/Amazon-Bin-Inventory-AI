"""
Amazon Bin Inventory AI - Model Evaluation & Performance Analytics
Computes test accuracy, MAE, confusion matrix, and inventory mismatch audit metrics.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch


from src.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEVICE,
    NUM_CLASSES,
    OUTPUTS_DIR,
)
from src.dataset import get_dataloaders
from src.model import load_trained_model
from src.utils import plot_confusion_matrix


def evaluate_test_set(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device = DEVICE,
) -> Dict:
    """
    Runs comprehensive evaluation on the test dataset split.
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_expected = []
    all_sample_ids = []

    print("🔍 Running evaluation on test set...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            expected = batch["expected_quantity"].to(device)
            sample_ids = batch["sample_id"]

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_expected.extend(expected.cpu().numpy().tolist())
            all_sample_ids.extend(sample_ids)

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)
    y_exp = np.array(all_expected)

    # 1. Vision Classification Metrics
    accuracy = float((y_pred == y_true).mean())
    mae = float(np.abs(y_pred - y_true).mean())
    rmse = float(np.sqrt(((y_pred - y_true) ** 2).mean()))

    # Per-class report
    present_classes = sorted(list(set(y_true) | set(y_pred)))
    target_names = [CLASS_NAMES[i] for i in present_classes]
    clf_report = classification_report(
        y_true, y_pred, labels=present_classes, target_names=target_names, output_dict=True, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cm_path = plot_confusion_matrix(cm, class_names=CLASS_NAMES)

    # 2. Inventory Mismatch Audit Metrics
    actual_mismatch = (y_true != y_exp)
    pred_mismatch = (y_pred != y_exp)

    mismatch_accuracy = float((actual_mismatch == pred_mismatch).mean())
    true_positives = int(np.logical_and(actual_mismatch, pred_mismatch).sum())
    false_positives = int(np.logical_and(~actual_mismatch, pred_mismatch).sum())
    false_negatives = int(np.logical_and(actual_mismatch, ~pred_mismatch).sum())
    true_negatives = int(np.logical_and(~actual_mismatch, ~pred_mismatch).sum())

    mismatch_precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    mismatch_recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    false_alarm_rate = (
        false_positives / (false_positives + true_negatives)
        if (false_positives + true_negatives) > 0
        else 0.0
    )

    results = {
        "num_test_samples": len(y_true),
        "accuracy": accuracy,
        "mae": mae,
        "rmse": rmse,
        "mismatch_audit": {
            "mismatch_accuracy": mismatch_accuracy,
            "precision": mismatch_precision,
            "recall": mismatch_recall,
            "false_alarm_rate": false_alarm_rate,
            "total_actual_mismatches": int(actual_mismatch.sum()),
            "total_detected_mismatches": int(pred_mismatch.sum()),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        },
        "confusion_matrix_path": str(cm_path),
        "classification_report": clf_report,
    }

    # Save results to json
    results_path = OUTPUTS_DIR / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 55)
    print("🎯 EVALUATION RESULTS SUMMARY")
    print("=" * 55)
    print(f"Total Test Samples:          {len(y_true)}")
    print(f"Top-1 Classification Acc:     {accuracy * 100:.2f}%")
    print(f"Mean Absolute Error (MAE):   {mae:.3f} items")
    print(f"Root Mean Squared Error:     {rmse:.3f} items")
    print("-" * 55)
    print("📦 INVENTORY MISMATCH AUDIT PERFORMANCE:")
    print(f"Audit Detection Accuracy:    {mismatch_accuracy * 100:.2f}%")
    print(f"Discrepancy Precision:       {mismatch_precision * 100:.2f}%")
    print(f"Discrepancy Recall:          {mismatch_recall * 100:.2f}%")
    print(f"False Alarm Rate:            {false_alarm_rate * 100:.2f}%")
    print("=" * 55)
    print(f"💾 Full evaluation metrics saved to: {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Amazon Bin Inventory AI model")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(BEST_MODEL_PATH),
        help="Path to trained model weights",
    )
    args = parser.parse_args()

    _, _, test_loader = get_dataloaders()
    model = load_trained_model(weights_path=args.model_path)
    evaluate_test_set(model, test_loader)
