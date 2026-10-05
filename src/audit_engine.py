"""
Amazon Bin Inventory AI - Inventory Audit & Mismatch Engine
Reconciles computer vision count predictions with warehouse metadata to flag discrepancies.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
import torch
from torchvision import transforms


from src.config import (
    CLASS_NAMES,
    DEVICE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_class_index,
    get_class_label,
)
from src.model import BinInventoryClassifier


# Inference image transform
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class InventoryAuditEngine:
    """
    Core auditing engine that executes vision-based inventory verification.
    """

    def __init__(self, model: BinInventoryClassifier, device: torch.device = DEVICE):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def predict_image(
        self, image: Union[Image.Image, Path, str]
    ) -> Tuple[int, float, Dict[str, float]]:
        """
        Runs visual inference on a PIL Image or image file path.
        Returns:
            predicted_count: estimated number of items (0 to 5)
            confidence: model probability for the predicted class
            probabilities: dict of class labels to probability floats
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_class_idx = int(np.argmax(probs))
        confidence = float(probs[pred_class_idx])

        # Map probabilities to class names
        prob_dict = {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        }

        # Count integer (0 to 5)
        predicted_count = pred_class_idx

        return predicted_count, confidence, prob_dict

    def parse_metadata(self, metadata_input: Union[Path, str, Dict]) -> Dict[str, Any]:
        """
        Parses Amazon metadata JSON from file path, raw string, or dictionary.
        """
        if isinstance(metadata_input, (str, Path)) and Path(metadata_input).exists():
            with open(metadata_input, "r", encoding="utf-8") as f:
                meta = json.load(f)
        elif isinstance(metadata_input, str):
            try:
                meta = json.loads(metadata_input)
            except Exception:
                meta = {}
        elif isinstance(metadata_input, dict):
            meta = metadata_input
        else:
            meta = {}

        expected_qty = int(meta.get("EXPECTED_QUANTITY", 0))
        bin_info = meta.get("bin_info", {})
        raw_skus = meta.get("BIN_FCSKU_DATA", {})

        # Extract structured SKU list
        skus_list = []
        if isinstance(raw_skus, dict):
            for asin, details in raw_skus.items():
                if isinstance(details, dict):
                    skus_list.append({
                        "asin": details.get("asin", asin),
                        "name": details.get("name", "Unknown Product"),
                        "product_type": details.get("product_type", "General"),
                        "dimensions": details.get("dimensions", {}),
                        "weight": details.get("weight", {}),
                    })
                else:
                    skus_list.append({"asin": asin, "name": str(details)})
        elif isinstance(raw_skus, list):
            for item in raw_skus:
                if isinstance(item, dict):
                    skus_list.append(item)

        return {
            "expected_quantity": expected_qty,
            "bin_info": bin_info,
            "skus": skus_list,
            "raw_metadata": meta,
        }

    def audit_bin(
        self,
        image: Union[Image.Image, Path, str],
        metadata_input: Union[Path, str, Dict],
    ) -> Dict[str, Any]:
        """
        Performs an end-to-end inventory audit of a single bin:
        1. Predicts visual item count
        2. Compares against recorded expected quantity
        3. Formulates discrepancy verdict, severity, and operational recommendations
        """
        # Step 1: Vision prediction
        pred_count, confidence, prob_dict = self.predict_image(image)

        # Step 2: Metadata extraction
        parsed_meta = self.parse_metadata(metadata_input)
        expected_qty = parsed_meta["expected_quantity"]

        # Step 3: Discrepancy analysis
        count_difference = pred_count - expected_qty

        if count_difference == 0:
            status = "MATCHED"
            severity = "success"
            verdict_title = "Inventory Verified (Matched)"
            verdict_message = (
                f"Visual count ({pred_count}) matches warehouse record ({expected_qty}). "
                "Bin is ready for downstream picking/stowing."
            )
            action_required = "None. Verified by AI vision inspection."
        elif count_difference < 0:
            status = "SHORTAGE"
            severity = "danger"
            verdict_title = "Discrepancy: Shortage Detected"
            shortage_amount = abs(count_difference)
            verdict_message = (
                f"Visual count ({pred_count}) is less than recorded quantity ({expected_qty}). "
                f"Missing {shortage_amount} item(s)."
            )
            action_required = (
                "Trigger Physical Audit: Check tote for dropped items, verify pick history, "
                "or report inventory shrinkage."
            )
        else:
            status = "OVERAGE"
            severity = "warning"
            verdict_title = "Discrepancy: Overage Detected"
            verdict_message = (
                f"Visual count ({pred_count}) exceeds recorded quantity ({expected_qty}). "
                f"Detected {count_difference} extra/unrecorded item(s)."
            )
            action_required = (
                "Trigger Physical Audit: Check for misplaced items from neighboring stowing tasks."
            )

        return {
            "status": status,
            "severity": severity,
            "verdict_title": verdict_title,
            "verdict_message": verdict_message,
            "action_required": action_required,
            "predicted_count": pred_count,
            "expected_quantity": expected_qty,
            "count_difference": count_difference,
            "confidence": confidence,
            "probabilities": prob_dict,
            "skus": parsed_meta["skus"],
            "bin_info": parsed_meta["bin_info"],
        }
