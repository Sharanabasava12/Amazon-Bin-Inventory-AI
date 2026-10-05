from pathlib import Path
import sys
import json

import cv2
import numpy as np
import pandas as pd

from skimage.feature import hog

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

IMAGE_SIZE = (128, 128)


# --------------------------------------------------
# HOG FEATURE EXTRACTION
# --------------------------------------------------

def extract_hog(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, IMAGE_SIZE)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features.astype(np.float32)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_dataset(csv_path):

    df = pd.read_csv(csv_path)

    features = []
    labels = []

    print(
        f"Extracting features from {len(df)} images..."
    )

    for i, row in df.iterrows():

        image_path = ROOT / row["image_path"]

        feature = extract_hog(image_path)

        features.append(feature)
        labels.append(int(row["label"]))

        if (i + 1) % 25 == 0:
            print(
                f"Processed {i + 1}/{len(df)}"
            )

    return (
        np.array(features),
        np.array(labels)
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("================================")
    print("RANDOM FOREST IMAGE CLASSIFICATION")
    print("================================")

    train_csv = ROOT / "data" / "processed" / "train.csv"
    val_csv = ROOT / "data" / "processed" / "val.csv"
    test_csv = ROOT / "data" / "processed" / "test.csv"

    # Load datasets
    X_train, y_train = load_dataset(train_csv)

    X_val, y_val = load_dataset(val_csv)

    X_test, y_test = load_dataset(test_csv)

    print("\nFeature shape:")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    # Combine training + validation
    X_train_final = np.concatenate(
        [X_train, X_val],
        axis=0
    )

    y_train_final = np.concatenate(
        [y_train, y_val],
        axis=0
    )

    print(
        "\nFinal training samples:",
        len(y_train_final)
    )

    # --------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_final,
        y_train_final
    )

    print("Random Forest training complete.")

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3, 4, 5]
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3, 4, 5],
        output_dict=True,
        zero_division=0
    )

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    print("\n==============================")
    print("RANDOM FOREST RESULTS")
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

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2, 3, 4, 5],
            zero_division=0
        )
    )

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------

    models_dir = ROOT / "models"

    models_dir.mkdir(
        exist_ok=True
    )

    model_path = (
        models_dir /
        "random_forest_hog.pkl"
    )

    import joblib

    joblib.dump(
        model,
        model_path
    )

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    outputs_dir = ROOT / "outputs"

    outputs_dir.mkdir(
        exist_ok=True
    )

    results = {
        "model": "Random Forest + HOG",
        "accuracy": float(accuracy),
        "rmse": float(rmse),
        "macro_precision": float(
            report["macro avg"]["precision"]
        ),
        "macro_recall": float(
            report["macro avg"]["recall"]
        ),
        "macro_f1": float(
            report["macro avg"]["f1-score"]
        ),
        "weighted_precision": float(
            report["weighted avg"]["precision"]
        ),
        "weighted_recall": float(
            report["weighted avg"]["recall"]
        ),
        "weighted_f1": float(
            report["weighted avg"]["f1-score"]
        ),
        "confusion_matrix": cm.tolist(),
        "test_samples": int(len(y_test))
    }

    results_path = (
        outputs_dir /
        "random_forest_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print("\nModel saved to:")
    print(model_path)

    print("\nResults saved to:")
    print(results_path)


if __name__ == "__main__":
    main()