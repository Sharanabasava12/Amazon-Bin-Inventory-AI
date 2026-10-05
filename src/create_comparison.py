from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

files = {
    "ResNet-34": OUTPUTS / "resnet34_results.json",
    "SVM + HOG": OUTPUTS / "svm_results.json",
    "Random Forest + HOG": OUTPUTS / "random_forest_results.json",
}

rows = []

for model, path in files.items():

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cm = data["confusion_matrix"]

    # Reconstruct true and predicted labels
    y_true = []
    y_pred = []

    for true_class, row in enumerate(cm):
        for predicted_class, count in enumerate(row):
            for _ in range(count):
                y_true.append(true_class)
                y_pred.append(predicted_class)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4, 5],
        average="macro",
        zero_division=0
    )

    rows.append({
        "Model": model,
        "Accuracy": data["accuracy"],
        "RMSE": data["rmse"],
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1
    })

df = pd.DataFrame(rows)

path = OUTPUTS / "model_comparison.csv"

df.to_csv(path, index=False)

print("\n================ MODEL COMPARISON ================\n")

print(
    df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "RMSE": "{:.4f}".format,
            "Macro Precision": "{:.4f}".format,
            "Macro Recall": "{:.4f}".format,
            "Macro F1": "{:.4f}".format,
        }
    )
)

print("\nSaved to:")
print(path)