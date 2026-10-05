import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from skimage.feature import hog

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    classification_report,
    confusion_matrix
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

IMAGE_SIZE = (128, 128)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# HOG Feature Extraction
# ---------------------------------------------------------


def extract_hog(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

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

# ---------------------------------------------------------
# Load CSV
# ---------------------------------------------------------

def load_dataset(csv_path):

    df = pd.read_csv(csv_path)

    features = []
    labels = []

    print(
        f"Extracting features from "
        f"{len(df)} images..."
    )

    for i, row in df.iterrows():

        image_path = Path(row["image_path"])

        feature = extract_hog(
            image_path
        )

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


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_model(
    model,
    X,
    y,
    name
):

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions
        )
    )

    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1, 2, 3, 4, 5]
    )

    report = classification_report(
        y,
        predictions,
        labels=[0, 1, 2, 3, 4, 5],
        zero_division=0
    )

    print("\n==============================")
    print(f"{name} RESULTS")
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

    return {
        "model": name,
        "accuracy": float(accuracy),
        "rmse": float(rmse),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "true_labels": y.tolist(),
        "predicted_labels": predictions.tolist()
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("================================")
    print("SVM IMAGE CLASSIFICATION")
    print("================================")

    train_csv = (
        PROCESSED_DIR / "train.csv"
    )

    val_csv = (
        PROCESSED_DIR / "val.csv"
    )

    test_csv = (
        PROCESSED_DIR / "test.csv"
    )

    # -----------------------------------------------------
    # Load datasets
    # -----------------------------------------------------

    X_train, y_train = load_dataset(
        train_csv
    )

    X_val, y_val = load_dataset(
        val_csv
    )

    X_test, y_test = load_dataset(
        test_csv
    )

    print("\nFeature shape:")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    # -----------------------------------------------------
    # Combine train + validation
    # -----------------------------------------------------
    #
    # We use validation for model selection.
    # After selecting the SVM configuration, retrain
    # using train + validation before final test.
    # -----------------------------------------------------

    X_train_full = np.vstack(
        [X_train, X_val]
    )

    y_train_full = np.concatenate(
        [y_train, y_val]
    )

    print(
        "\nFinal training samples:",
        len(y_train_full)
    )

    # -----------------------------------------------------
    # SVM model
    # -----------------------------------------------------

    print("\nTraining SVM...")

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        random_state=42
    )

    model.fit(
        X_train_full,
        y_train_full
    )

    print("SVM training complete.")

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    results = evaluate_model(
        model,
        X_test,
        y_test,
        "SVM"
    )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    model_path = (
        MODELS_DIR / "svm_hog.pkl"
    )

    import joblib

    joblib.dump(
        model,
        model_path
    )

    print(
        f"\nModel saved to:\n{model_path}"
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    results_path = (
        OUTPUT_DIR / "svm_results.json"
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

    print(
        f"Results saved to:\n{results_path}"
    )


if __name__ == "__main__":
    main()