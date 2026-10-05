from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

comparison_file = OUTPUTS / "model_comparison.csv"


# ============================================================
# LOAD COMPARISON DATA
# ============================================================

if not comparison_file.exists():
    raise FileNotFoundError(
        "model_comparison.csv not found. "
        "Run: python src\\create_comparison.py"
    )

df = pd.read_csv(comparison_file)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(df.to_string(index=False))


# ============================================================
# 1. ACCURACY COMPARISON
# ============================================================

plt.figure(figsize=(8, 5))

values = df["Accuracy"] * 100

plt.bar(df["Model"], values)

plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title("Model Accuracy Comparison")

plt.ylim(0, 100)

for i, value in enumerate(values):
    plt.text(
        i,
        value + 2,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    OUTPUTS / "accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. RMSE COMPARISON
# ============================================================

plt.figure(figsize=(8, 5))

values = df["RMSE"]

plt.bar(df["Model"], values)

plt.ylabel("RMSE")
plt.xlabel("Model")
plt.title("Model RMSE Comparison")

for i, value in enumerate(values):
    plt.text(
        i,
        value + 0.03,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    OUTPUTS / "rmse_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 3. MACRO F1 COMPARISON
# ============================================================

plt.figure(figsize=(8, 5))

values = df["Macro F1"]

plt.bar(df["Model"], values)

plt.ylabel("Macro F1")
plt.xlabel("Model")
plt.title("Macro F1 Comparison")

plt.ylim(0, max(values) + 0.1)

for i, value in enumerate(values):
    plt.text(
        i,
        value + 0.01,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    OUTPUTS / "f1_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 4. PRECISION / RECALL / F1 COMPARISON
# ============================================================

x = np.arange(len(df))
width = 0.25

plt.figure(figsize=(9, 5))

plt.bar(
    x - width,
    df["Macro Precision"],
    width,
    label="Precision"
)

plt.bar(
    x,
    df["Macro Recall"],
    width,
    label="Recall"
)

plt.bar(
    x + width,
    df["Macro F1"],
    width,
    label="F1"
)

plt.xticks(
    x,
    df["Model"]
)

plt.ylabel("Score")
plt.xlabel("Model")
plt.title("Precision, Recall and F1 Comparison")

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUTS / "precision_recall_f1_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. CONFUSION MATRICES
# ============================================================

result_files = {
    "ResNet-34": OUTPUTS / "resnet34_results.json",
    "SVM + HOG": OUTPUTS / "svm_results.json",
    "Random Forest + HOG": OUTPUTS / "random_forest_results.json",
}

class_names = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5+"
]


for model_name, file_path in result_files.items():

    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cm = np.array(data["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(7, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        values_format="d"
    )

    ax.set_title(
        f"{model_name} Confusion Matrix"
    )

    plt.tight_layout()

    safe_name = (
        model_name
        .lower()
        .replace(" + ", "_")
        .replace(" ", "_")
        .replace("-", "")
    )

    output_file = (
        OUTPUTS /
        f"confusion_matrix_{safe_name}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Created: {output_file}")


# ============================================================
# FINISHED
# ============================================================

print("\n==============================")
print("VISUALIZATIONS CREATED")
print("==============================")

print("Created:")
print(" - accuracy_comparison.png")
print(" - rmse_comparison.png")
print(" - f1_comparison.png")
print(" - precision_recall_f1_comparison.png")
print(" - confusion_matrix_resnet34.png")
print(" - confusion_matrix_svm_hog.png")
print(" - confusion_matrix_random_forest_hog.png")

print("\nAll visualizations completed successfully.")