from pathlib import Path
import json

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"

RESNET_MODEL = MODEL_DIR / "resnet34_best.pth"
BATCH_RESULTS = OUTPUT_DIR / "batch_audit_results.csv"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Amazon Bin Inventory AI",
    page_icon="📦",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #777;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #ddd;
        text-align: center;
    }

    .status-match {
        padding: 15px;
        border-radius: 10px;
        background-color: #dff5df;
        color: #176b17;
        font-size: 22px;
        font-weight: 700;
        text-align: center;
    }

    .status-shortage {
        padding: 15px;
        border-radius: 10px;
        background-color: #ffdede;
        color: #a00000;
        font-size: 22px;
        font-weight: 700;
        text-align: center;
    }

    .status-overage {
        padding: 15px;
        border-radius: 10px;
        background-color: #fff0c2;
        color: #795600;
        font-size: 22px;
        font-weight: 700;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL SETTINGS
# ============================================================

NUM_CLASSES = 6

CLASS_NAMES = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5+"
}


# ============================================================
# LOAD RESNET-34
# ============================================================

@st.cache_resource
def load_resnet_model():

    model = models.resnet34(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    if not RESNET_MODEL.exists():

        raise FileNotFoundError(
            f"Model not found: {RESNET_MODEL}"
        )

    checkpoint = torch.load(
        RESNET_MODEL,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.eval()

    return model


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ]
)


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image):

    model = load_resnet_model()

    image = image.convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    with torch.no_grad():

        outputs = model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = prediction.item()

    confidence_value = confidence.item()

    return (
        predicted_class,
        confidence_value,
        probabilities[0].numpy()
    )


# ============================================================
# AUDIT ENGINE
# ============================================================

def audit_inventory(
    predicted_quantity,
    expected_quantity
):

    difference = (
        predicted_quantity -
        expected_quantity
    )

    if difference == 0:

        status = "MATCHED"

    elif difference < 0:

        status = "SHORTAGE"

    else:

        status = "OVERAGE"

    return status, difference


# ============================================================
# LOAD MODEL METRICS
# ============================================================

@st.cache_data
def load_metrics():

    file = OUTPUT_DIR / "model_comparison.csv"

    if not file.exists():

        return None

    return pd.read_csv(file)


# ============================================================
# LOAD BATCH AUDIT RESULTS
# ============================================================

@st.cache_data
def load_batch_results():

    if not BATCH_RESULTS.exists():

        return None

    try:

        df = pd.read_csv(BATCH_RESULTS)

        return df

    except Exception as e:

        st.error(
            f"Could not read batch audit results: {e}"
        )

        return None


# ============================================================
# LOAD DATASET SUMMARY
# ============================================================

@st.cache_data
def load_dataset_summary():

    summary_file = (
        OUTPUT_DIR /
        "dataset_summary.json"
    )

    if not summary_file.exists():

        return None

    try:

        with open(
            summary_file,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📦 Navigation")

page = st.sidebar.radio(
    "Select module",
    [
        "Single Bin Inspector",
        "Batch Inventory Auditor",
        "Model Performance",
        "Dataset Information"
    ]
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '📦 Amazon Bin Inventory AI'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Computer Vision Based Warehouse Inventory Audit System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PAGE 1
# SINGLE BIN INSPECTOR
# ============================================================

if page == "Single Bin Inspector":

    st.header("🔍 Single Bin Inspector")

    st.write(
        "Upload an Amazon bin image and compare the "
        "visual quantity prediction with the recorded "
        "warehouse quantity."
    )

    col1, col2 = st.columns(2)

    with col1:

        uploaded_file = st.file_uploader(
            "Upload Bin Image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )

    with col2:

        expected_quantity = st.number_input(
            "Recorded Expected Quantity",
            min_value=0,
            max_value=19,
            value=1,
            step=1
        )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        )

        st.subheader("Bin Image")

        st.image(
            image,
            use_container_width=True
        )

        if st.button(
            "🔎 Analyze Bin",
            type="primary"
        ):

            try:

                with st.spinner(
                    "Running ResNet-34 prediction..."
                ):

                    (
                        prediction,
                        confidence,
                        probabilities
                    ) = predict_image(image)

                # ----------------------------------------
                # Convert prediction
                # ----------------------------------------

                predicted_quantity = prediction

                if prediction == 5:

                    display_prediction = "5+"

                    numeric_prediction = 5

                else:

                    display_prediction = str(
                        prediction
                    )

                    numeric_prediction = prediction

                # ----------------------------------------
                # Audit
                # ----------------------------------------

                status, difference = audit_inventory(
                    numeric_prediction,
                    expected_quantity
                )

                # ----------------------------------------
                # Results
                # ----------------------------------------

                st.divider()

                st.subheader(
                    "📊 Inspection Result"
                )

                c1, c2, c3, c4 = st.columns(4)

                with c1:

                    st.metric(
                        "Vision Count",
                        display_prediction
                    )

                with c2:

                    st.metric(
                        "Expected Quantity",
                        expected_quantity
                    )

                with c3:

                    st.metric(
                        "Difference",
                        difference
                    )

                with c4:

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                # ----------------------------------------
                # Verdict
                # ----------------------------------------

                st.subheader(
                    "Audit Verdict"
                )

                if status == "MATCHED":

                    st.markdown(
                        '<div class="status-match">'
                        '🟢 MATCHED'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.success(
                        "Visual prediction matches the "
                        "recorded inventory quantity."
                    )

                elif status == "SHORTAGE":

                    st.markdown(
                        '<div class="status-shortage">'
                        '🔴 SHORTAGE'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.warning(
                        "The predicted physical quantity "
                        "is lower than the recorded quantity."
                    )

                else:

                    st.markdown(
                        '<div class="status-overage">'
                        '🟡 OVERAGE'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.warning(
                        "The predicted physical quantity "
                        "is higher than the recorded quantity."
                    )

                # ----------------------------------------
                # Probability chart
                # ----------------------------------------

                st.subheader(
                    "Prediction Probability Distribution"
                )

                probability_df = pd.DataFrame(
                    {
                        "Quantity": [
                            "0",
                            "1",
                            "2",
                            "3",
                            "4",
                            "5+"
                        ],
                        "Probability": probabilities
                    }
                )

                st.bar_chart(
                    probability_df.set_index(
                        "Quantity"
                    )
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )


# ============================================================
# PAGE 2
# BATCH INVENTORY AUDITOR
# ============================================================

elif page == "Batch Inventory Auditor":

    st.header(
        "📋 Batch Inventory Auditor"
    )

    st.write(
        "Analyze all Amazon bin images automatically "
        "and identify inventory matches, shortages, "
        "and overages."
    )

    st.divider()

    # ========================================================
    # LOAD DATA
    # ========================================================

    batch_df = load_batch_results()

    if batch_df is None:

        st.error(
            "Batch audit results were not found."
        )

        st.info(
            "Run the following command first:"
        )

        st.code(
            "python src\\batch_auditor.py",
            language="powershell"
        )

    else:

        # ====================================================
        # CLEAN DATA
        # ====================================================

        required_columns = [
            "bin_id",
            "image",
            "expected_quantity",
            "predicted_quantity",
            "confidence",
            "difference",
            "status"
        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in batch_df.columns
        ]

        if missing_columns:

            st.error(
                "Missing columns in batch audit CSV: "
                + ", ".join(missing_columns)
            )

        else:

            # Ensure correct types

            batch_df[
                "expected_quantity"
            ] = pd.to_numeric(
                batch_df[
                    "expected_quantity"
                ],
                errors="coerce"
            )

            batch_df[
                "predicted_quantity"
            ] = pd.to_numeric(
                batch_df[
                    "predicted_quantity"
                ],
                errors="coerce"
            )

            batch_df[
                "confidence"
            ] = pd.to_numeric(
                batch_df[
                    "confidence"
                ],
                errors="coerce"
            )

            batch_df[
                "difference"
            ] = pd.to_numeric(
                batch_df[
                    "difference"
                ],
                errors="coerce"
            )

            # =================================================
            # KPIs
            # =================================================

            total_bins = len(batch_df)

            matched = int(
                (
                    batch_df["status"]
                    == "MATCHED"
                ).sum()
            )

            shortages = int(
                (
                    batch_df["status"]
                    == "SHORTAGE"
                ).sum()
            )

            overages = int(
                (
                    batch_df["status"]
                    == "OVERAGE"
                ).sum()
            )

            errors = total_bins - (
                matched +
                shortages +
                overages
            )

            match_rate = (
                matched / total_bins * 100
                if total_bins > 0
                else 0
            )

            shortage_rate = (
                shortages / total_bins * 100
                if total_bins > 0
                else 0
            )

            overage_rate = (
                overages / total_bins * 100
                if total_bins > 0
                else 0
            )

            # =================================================
            # KPI CARDS
            # =================================================

            st.subheader(
                "📊 Warehouse Audit Summary"
            )

            k1, k2, k3, k4 = st.columns(4)

            with k1:

                st.metric(
                    "Total Bins",
                    total_bins
                )

            with k2:

                st.metric(
                    "🟢 Matched",
                    matched,
                    f"{match_rate:.2f}%"
                )

            with k3:

                st.metric(
                    "🔴 Shortages",
                    shortages,
                    f"{shortage_rate:.2f}%"
                )

            with k4:

                st.metric(
                    "🟡 Overages",
                    overages,
                    f"{overage_rate:.2f}%"
                )

            st.divider()

            # =================================================
            # STATUS SUMMARY
            # =================================================

            st.subheader(
                "📈 Audit Status Distribution"
            )

            status_df = pd.DataFrame(
                {
                    "Status": [
                        "MATCHED",
                        "SHORTAGE",
                        "OVERAGE"
                    ],
                    "Bins": [
                        matched,
                        shortages,
                        overages
                    ]
                }
            )

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:

                st.bar_chart(
                    status_df.set_index(
                        "Status"
                    )
                )

            with chart_col2:

                st.dataframe(
                    status_df,
                    use_container_width=True,
                    hide_index=True
                )

            st.divider()

            # =================================================
            # FILTERS
            # =================================================

            st.subheader(
                "🔎 Filter Audit Results"
            )

            f1, f2, f3 = st.columns(3)

            with f1:

                selected_status = st.multiselect(
                    "Status",
                    options=[
                        "MATCHED",
                        "SHORTAGE",
                        "OVERAGE"
                    ],
                    default=[
                        "MATCHED",
                        "SHORTAGE",
                        "OVERAGE"
                    ]
                )

            with f2:

                min_confidence = st.slider(
                    "Minimum Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05
                )

            with f3:

                show_only_discrepancies = st.checkbox(
                    "Show only discrepancies"
                )

            filtered_df = batch_df.copy()

            if selected_status:

                filtered_df = filtered_df[
                    filtered_df["status"].isin(
                        selected_status
                    )
                ]

            filtered_df = filtered_df[
                filtered_df["confidence"]
                >= min_confidence
            ]

            if show_only_discrepancies:

                filtered_df = filtered_df[
                    filtered_df["status"]
                    != "MATCHED"
                ]

            # =================================================
            # FILTERED RESULTS
            # =================================================

            st.write(
                f"Showing **{len(filtered_df)}** "
                f"of **{total_bins}** bins."
            )

            display_df = filtered_df.copy()

            display_df[
                "confidence"
            ] = (
                display_df[
                    "confidence"
                ] * 100
            ).round(2)

            display_df = display_df.rename(
                columns={
                    "bin_id": "Bin ID",
                    "image": "Image",
                    "expected_quantity": "Expected",
                    "predicted_quantity": "Predicted",
                    "confidence": "Confidence (%)",
                    "difference": "Difference",
                    "status": "Status"
                }
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            # =================================================
            # DOWNLOAD
            # =================================================

            st.subheader(
                "📥 Export Audit Report"
            )

            csv_data = filtered_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Filtered Audit CSV",
                data=csv_data,
                file_name="amazon_bin_audit_report.csv",
                mime="text/csv"
            )

            # =================================================
            # DISCREPANCY SECTION
            # =================================================

            st.divider()

            st.subheader(
                "🚨 Inventory Discrepancies"
            )

            discrepancy_df = batch_df[
                batch_df["status"]
                != "MATCHED"
            ].copy()

            if len(discrepancy_df) == 0:

                st.success(
                    "No inventory discrepancies detected."
                )

            else:

                d1, d2 = st.columns(2)

                with d1:

                    st.error(
                        f"🔴 {shortages} bins have potential shortages."
                    )

                with d2:

                    st.warning(
                        f"🟡 {overages} bins have potential overages."
                    )

                st.dataframe(
                    discrepancy_df.rename(
                        columns={
                            "bin_id": "Bin ID",
                            "image": "Image",
                            "expected_quantity": "Expected",
                            "predicted_quantity": "Predicted",
                            "confidence": "Confidence",
                            "difference": "Difference",
                            "status": "Status"
                        }
                    ),
                    use_container_width=True,
                    hide_index=True
                )

            # =================================================
            # OPERATIONAL INTERPRETATION
            # =================================================

            st.divider()

            st.subheader(
                "🏭 Operational Interpretation"
            )

            st.write(
                f"""
                **Total bins scanned:** {total_bins}

                **Matched bins:** {matched} "
                ({match_rate:.2f}%)

                **Potential shortages:** {shortages} "
                ({shortage_rate:.2f}%)

                **Potential overages:** {overages} "
                ({overage_rate:.2f}%)
                """
            )

            st.info(
                "MATCHED bins indicate that the predicted "
                "visual quantity equals the recorded quantity. "
                "SHORTAGE bins may indicate missing inventory, "
                "picking errors, or inventory shrinkage. "
                "OVERAGE bins may indicate unrecorded stock "
                "or misplaced inventory."
            )

            if errors > 0:

                st.warning(
                    f"{errors} records contain an unexpected "
                    "status and should be inspected."
                )


# ============================================================
# PAGE 3
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header(
        "📈 Model Performance"
    )

    metrics = load_metrics()

    if metrics is None:

        st.error(
            "model_comparison.csv not found."
        )

    else:

        st.subheader(
            "Model Comparison"
        )

        display_df = metrics.copy()

        display_df["Accuracy"] = (
            display_df["Accuracy"] * 100
        ).round(2)

        display_df["RMSE"] = (
            display_df["RMSE"]
            .round(4)
        )

        display_df["Macro Precision"] = (
            display_df["Macro Precision"]
            .round(4)
        )

        display_df["Macro Recall"] = (
            display_df["Macro Recall"]
            .round(4)
        )

        display_df["Macro F1"] = (
            display_df["Macro F1"]
            .round(4)
        )

        display_df = display_df.rename(
            columns={
                "Accuracy": "Accuracy (%)"
            }
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Accuracy Comparison"
            )

            accuracy_file = (
                OUTPUT_DIR /
                "accuracy_comparison.png"
            )

            if accuracy_file.exists():

                st.image(
                    accuracy_file,
                    use_container_width=True
                )

        with col2:

            st.subheader(
                "RMSE Comparison"
            )

            rmse_file = (
                OUTPUT_DIR /
                "rmse_comparison.png"
            )

            if rmse_file.exists():

                st.image(
                    rmse_file,
                    use_container_width=True
                )

        st.subheader(
            "Macro F1 Comparison"
        )

        f1_file = (
            OUTPUT_DIR /
            "f1_comparison.png"
        )

        if f1_file.exists():

            st.image(
                f1_file,
                use_container_width=True
            )

        st.subheader(
            "Precision / Recall / F1"
        )

        pr_file = (
            OUTPUT_DIR /
            "precision_recall_f1_comparison.png"
        )

        if pr_file.exists():

            st.image(
                pr_file,
                use_container_width=True
            )

        st.subheader(
            "Confusion Matrices"
        )

        c1, c2, c3 = st.columns(3)

        resnet_cm = (
            OUTPUT_DIR /
            "confusion_matrix_resnet34.png"
        )

        svm_cm = (
            OUTPUT_DIR /
            "confusion_matrix_svm_hog.png"
        )

        rf_cm = (
            OUTPUT_DIR /
            "confusion_matrix_random_forest_hog.png"
        )

        with c1:

            if resnet_cm.exists():

                st.image(
                    resnet_cm,
                    caption="ResNet-34",
                    use_container_width=True
                )

        with c2:

            if svm_cm.exists():

                st.image(
                    svm_cm,
                    caption="SVM + HOG",
                    use_container_width=True
                )

        with c3:

            if rf_cm.exists():

                st.image(
                    rf_cm,
                    caption="Random Forest + HOG",
                    use_container_width=True
                )


# ============================================================
# PAGE 4
# DATASET INFORMATION
# ============================================================

elif page == "Dataset Information":

    st.header(
        "📊 Dataset Information"
    )

    summary = load_dataset_summary()

    if summary is not None:

        st.json(summary)

    else:

        st.info(
            "Dataset summary file not found."
        )

    st.subheader(
        "Dataset Files"
    )

    image_dir = DATA_DIR / "images"
    metadata_dir = DATA_DIR / "metadata"

    image_count = len(
        list(
            image_dir.glob("*.jpg")
        )
    ) if image_dir.exists() else 0

    metadata_count = len(
        list(
            metadata_dir.glob("*.json")
        )
    ) if metadata_dir.exists() else 0

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Images",
            image_count
        )

    with c2:

        st.metric(
            "Metadata Records",
            metadata_count
        )

    st.info(
        "The current experiment uses the filtered "
        "0–5+ quantity classification dataset."
    )

    # --------------------------------------------------------
    # Dataset split information
    # --------------------------------------------------------

    st.subheader(
        "📂 Dataset Splits"
    )

    train_file = (
        DATA_DIR /
        "processed" /
        "train.csv"
    )

    val_file = (
        DATA_DIR /
        "processed" /
        "val.csv"
    )

    test_file = (
        DATA_DIR /
        "processed" /
        "test.csv"
    )

    split_col1, split_col2, split_col3 = st.columns(3)

    with split_col1:

        if train_file.exists():

            train_df = pd.read_csv(
                train_file
            )

            st.metric(
                "Training Samples",
                len(train_df)
            )

    with split_col2:

        if val_file.exists():

            val_df = pd.read_csv(
                val_file
            )

            st.metric(
                "Validation Samples",
                len(val_df)
            )

    with split_col3:

        if test_file.exists():

            test_df = pd.read_csv(
                test_file
            )

            st.metric(
                "Test Samples",
                len(test_df)
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Amazon Bin Inventory AI | "
    "ResNet-34 + SVM + Random Forest Extension"
)