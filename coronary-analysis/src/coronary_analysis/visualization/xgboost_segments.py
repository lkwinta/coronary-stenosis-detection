"""Plotting helpers for XGBoost segment experiments."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon, Rectangle
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve


def parse_json_points(value):
    """Parse a JSON/list point array into an Nx2 numpy array, or return None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    return arr


def plot_precision_recall(y_true, y_score, average_precision: float | None = None):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    _, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    title = "XGBoost Precision-Recall"
    if average_precision is not None:
        title += f", AP={average_precision:.4f}"
    ax.set_title(title)
    ax.grid(True)
    return ax


def plot_roc(y_true, y_score, roc_auc: float | None = None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    _, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate / Recall")
    title = "XGBoost ROC"
    if roc_auc is not None:
        title += f", AUC={roc_auc:.4f}"
    ax.set_title(title)
    ax.grid(True)
    return ax


def plot_confusion_matrix(y_true, y_pred, title: str | None = None):
    _, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, values_format="d")
    if title:
        ax.set_title(title)
    return ax


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 30):
    top_imp = importance_df.head(top_n).iloc[::-1]
    _, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top_imp["feature"], top_imp["importance"])
    ax.set_xlabel("Importance")
    ax.set_title(f"XGBoost feature importance — top {top_n}")
    plt.tight_layout()
    return ax


def plot_xgb_predictions_for_image(
    predictions_df: pd.DataFrame,
    image_by_id: dict,
    anns_by_image: dict,
    image_dir: str | Path,
    image_id: int | None = None,
    threshold: float = 0.5,
    only_test: bool = True,
):
    """Visualize segment-level XGBoost predictions for one image."""
    plot_df = predictions_df.copy()
    if only_test:
        plot_df = plot_df[plot_df["split"] == "test"].copy()

    if image_id is None:
        candidates = plot_df.groupby("image_id")["label"].sum().sort_values(ascending=False)
        if len(candidates) == 0:
            raise ValueError("No data to visualize.")
        image_id = int(candidates.index[0])

    rec = image_by_id[int(image_id)]
    img_path = Path(image_dir) / rec["file_name"]
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)

    sub = plot_df[plot_df["image_id"].astype(int) == int(image_id)].copy()
    if len(sub) == 0:
        raise ValueError(f"No segments for image_id={image_id} in the selected split.")

    sub["pred"] = (sub["xgb_pred_proba"] >= threshold).astype(int)
    anns = anns_by_image.get(int(image_id), [])

    _, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap="gray")
    ax.set_title(
        f"XGBoost predictions, image_id={image_id}, file={rec['file_name']}\n"
        f"TP=red, TN=orange, FP=purple, FN=blue, threshold={threshold:.3f}"
    )
    ax.axis("off")

    for ann in anns:
        if "bbox" in ann and ann["bbox"] is not None:
            x, y, bw, bh = ann["bbox"]
            ax.add_patch(Rectangle((x, y), bw, bh, linewidth=2, edgecolor="black", facecolor="none"))
        seg = ann.get("segmentation")
        if isinstance(seg, list):
            for poly in seg:
                arr = np.asarray(poly, dtype=float).reshape(-1, 2)
                if len(arr) >= 3:
                    ax.add_patch(
                        Polygon(arr, closed=True, edgecolor="cyan", facecolor="cyan", alpha=0.2, linewidth=1.5)
                    )

    for _, row in sub.iterrows():
        box = parse_json_points(row["oriented_box_vertices_xy"])
        if box is None or len(box) < 4:
            continue

        true = int(row["label"])
        pred = int(row["pred"])
        if true == 1 and pred == 1:
            color, alpha, lw = "red", 0.45, 1.8
        elif true == 0 and pred == 0:
            color, alpha, lw = "orange", 0.16, 0.7
        elif true == 0 and pred == 1:
            color, alpha, lw = "purple", 0.45, 1.8
        else:
            color, alpha, lw = "blue", 0.55, 2.2
        ax.add_patch(Polygon(box, closed=True, edgecolor=color, facecolor=color, alpha=alpha, linewidth=lw))

    return ax, sub, pd.crosstab(sub["label"], sub["pred"], rownames=["true"], colnames=["pred"])
