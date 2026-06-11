"""XGBoost baseline for oriented coronary-vessel segments.

This module contains the reusable logic that used to live in the notebook:
feature engineering, grouped train/val/test split, model construction, training,
threshold selection, evaluation and prediction table creation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError("Install xgboost first, e.g. `uv add xgboost` or `pip install xgboost`.") from exc


DEFAULT_EXCLUDE_COLS: set[str] = {
    "label",
    "image_id",
    "file_name",
    "branch_id",
    "oriented_segment_id",
    "local_segment_id",
    "matched_annotation_ids",
    "matched_annotation_bboxes_xywh",
    "positive_patch_overlap_fraction",
    "positive_patch_intersection_area_px",
    "oriented_box_vertices_xy",
    "centerline_points_xy",
    "patch_pixels_xy",
    "used_junction_cleanup",
}

DEFAULT_NEIGHBOR_BASE_COLS: tuple[str, ...] = (
    "segment_length_px",
    "patch_width_px",
    "patch_area_px",
    "angle_deg",
    "mean_diameter",
    "min_diameter",
    "max_diameter",
    "std_diameter",
    "diameter_drop",
)


@dataclass(frozen=True)
class SegmentSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    seed: int


@dataclass
class XGBoostSegmentResult:
    model: XGBClassifier
    dataframe: pd.DataFrame
    feature_cols: list[str]
    split: SegmentSplit
    selected_threshold: float
    metrics: dict[str, object]
    predictions: pd.DataFrame
    importance: pd.DataFrame


@dataclass
class XGBoostSegmentArtifact:
    """Persisted XGBoost model bundle used by the analysis pipeline.

    It stores everything needed for deterministic inference: the fitted model,
    feature order and the probability threshold selected on validation data.
    """

    model: XGBClassifier
    feature_cols: list[str]
    threshold: float
    metadata: dict[str, object]


def prepare_binary_segments(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Keep only 0/1 labels and cast the target to int."""
    out = df.copy()
    out = out[out[label_col].isin([0, 1])].copy()
    out[label_col] = out[label_col].astype(int)

    if out[label_col].nunique() < 2:
        raise ValueError(
            "The segment dataframe contains only one class. Generate more samples or increase FAST_DEV_LIMIT."
        )
    return out


def add_neighbor_features(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("image_id", "branch_id"),
    sort_cols: Sequence[str] = ("image_id", "branch_id", "local_segment_id"),
    base_cols: Sequence[str] = DEFAULT_NEIGHBOR_BASE_COLS,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Add previous/next segment context on the same vessel branch."""
    missing_sort_cols = [col for col in sort_cols if col not in df.columns]
    if missing_sort_cols:
        raise KeyError(f"Missing columns needed to sort/group segments: {missing_sort_cols}")

    out = df.sort_values(list(sort_cols)).reset_index(drop=True).copy()
    base_cols = [col for col in base_cols if col in out.columns]

    for col in base_cols:
        out[f"prev_{col}"] = out.groupby(list(group_cols))[col].shift(1)
        out[f"next_{col}"] = out.groupby(list(group_cols))[col].shift(-1)
        out[f"neighbor_mean_{col}"] = out[[f"prev_{col}", f"next_{col}"]].mean(axis=1)
        out[f"diff_vs_prev_{col}"] = out[col] - out[f"prev_{col}"]
        out[f"diff_vs_next_{col}"] = out[col] - out[f"next_{col}"]
        out[f"diff_vs_neighbor_mean_{col}"] = out[col] - out[f"neighbor_mean_{col}"]

    if {"mean_diameter", "neighbor_mean_mean_diameter"}.issubset(out.columns):
        out["mean_diameter_ratio_to_neighbors"] = out["mean_diameter"] / (
            out["neighbor_mean_mean_diameter"] + eps
        )

    if {"min_diameter", "neighbor_mean_mean_diameter"}.issubset(out.columns):
        out["min_diameter_ratio_to_neighbor_mean"] = out["min_diameter"] / (
            out["neighbor_mean_mean_diameter"] + eps
        )

    if {"min_diameter", "prev_mean_diameter", "next_mean_diameter"}.issubset(out.columns):
        neighbor_mean = out[["prev_mean_diameter", "next_mean_diameter"]].mean(axis=1)
        out["local_stenosis_score"] = 1.0 - (out["min_diameter"] / (neighbor_mean + eps))

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)

    for col in out.columns:
        if col.startswith("diff_vs_"):
            out[col] = out[col].fillna(0)

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    medians = out[numeric_cols].median(numeric_only=True).fillna(0)
    out[numeric_cols] = out[numeric_cols].fillna(medians).fillna(0)
    return out


def select_feature_columns(
    df: pd.DataFrame,
    exclude_cols: Iterable[str] = DEFAULT_EXCLUDE_COLS,
) -> list[str]:
    """Select numeric model features while excluding leakage/debug columns."""
    exclude_cols = set(exclude_cols)
    return [
        col
        for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]


def grouped_train_val_test_split(
    data: pd.DataFrame,
    labels: pd.Series,
    group_ids: pd.Series,
    random_state: int = 42,
    test_size: float = 0.15,
    val_size: float = 0.15,
    max_tries: int = 100,
) -> SegmentSplit:
    """Split by image/group so segments from one image do not leak across splits."""
    last_split: tuple[np.ndarray, np.ndarray, np.ndarray, int] | None = None

    for seed in range(random_state, random_state + max_tries):
        first = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size + val_size,
            random_state=seed,
        )
        train_idx, temp_idx = next(first.split(data, labels, groups=group_ids))

        temp_group_ids = group_ids.iloc[temp_idx]
        temp_labels = labels.iloc[temp_idx]
        relative_test_size = test_size / (test_size + val_size)

        second = GroupShuffleSplit(
            n_splits=1,
            test_size=relative_test_size,
            random_state=seed + 1000,
        )
        val_rel_idx, test_rel_idx = next(
            second.split(data.iloc[temp_idx], temp_labels, groups=temp_group_ids)
        )

        val_idx = temp_idx[val_rel_idx]
        test_idx = temp_idx[test_rel_idx]
        last_split = (train_idx, val_idx, test_idx, seed)

        split_labels = {
            "train": labels.iloc[train_idx],
            "val": labels.iloc[val_idx],
            "test": labels.iloc[test_idx],
        }
        if all(part.nunique() == 2 for part in split_labels.values()):
            return SegmentSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, seed=seed)

    if last_split is None:
        raise RuntimeError("Could not create a grouped split.")

    train_idx, val_idx, test_idx, seed = last_split
    return SegmentSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, seed=seed)


def make_xgb_classifier(y_train: pd.Series, random_state: int = 42, **overrides) -> XGBClassifier:
    """Create the baseline XGBoost classifier with class imbalance handling."""
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    params = dict(
        n_estimators=800,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=2.0,
        reg_alpha=0.1,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )
    params.update(overrides)
    return XGBClassifier(**params)


def select_threshold_for_recall(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    target_recall: float = 0.85,
    fallback: float = 0.5,
) -> float:
    """Pick the validation threshold with max F1 among thresholds meeting target recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    valid = np.where(recall[:-1] >= target_recall)[0]
    if len(valid) == 0:
        return fallback

    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = valid[np.argmax(f1[valid])]
    return float(thresholds[best_idx])


def evaluate_binary_classifier(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    """Return scalar metrics, confusion matrix and a sklearn classification report."""
    y_true_series = pd.Series(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true_series, y_score) if y_true_series.nunique() == 2 else np.nan,
        "average_precision": average_precision_score(y_true_series, y_score)
        if y_true_series.nunique() == 2
        else np.nan,
        "confusion_matrix": confusion_matrix(y_true_series, y_pred),
        "classification_report": classification_report(y_true_series, y_pred, digits=4),
    }


def make_predictions_dataframe(
    df: pd.DataFrame,
    split: SegmentSplit,
    proba_val: np.ndarray,
    proba_test: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Attach split names and XGBoost probabilities/predictions to the segment dataframe."""
    out = df.copy()
    out["split"] = "unused"
    out.loc[out.index[split.train_idx], "split"] = "train"
    out.loc[out.index[split.val_idx], "split"] = "val"
    out.loc[out.index[split.test_idx], "split"] = "test"

    out["xgb_pred_proba"] = np.nan
    out.loc[out.index[split.val_idx], "xgb_pred_proba"] = proba_val
    out.loc[out.index[split.test_idx], "xgb_pred_proba"] = proba_test

    out["xgb_pred_label"] = np.nan
    out.loc[out.index[split.test_idx], "xgb_pred_label"] = (
        out.loc[out.index[split.test_idx], "xgb_pred_proba"] >= threshold
    ).astype(int)
    return out


def train_xgboost_on_segments(
    df: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.15,
    val_size: float = 0.15,
    target_recall: float = 0.85,
    xgb_overrides: dict | None = None,
    model_output_path: str | Path | None = None,
    fit_verbose: int | bool = 50,
) -> XGBoostSegmentResult:
    """Run the full reusable XGBoost baseline pipeline."""
    xgb_df = prepare_binary_segments(df)
    xgb_df = add_neighbor_features(xgb_df)
    feature_cols = select_feature_columns(xgb_df)

    x_all = xgb_df[feature_cols].copy()
    y_all = xgb_df["label"].astype(int).copy()
    groups = xgb_df["image_id"].copy()

    split = grouped_train_val_test_split(
        x_all,
        y_all,
        groups,
        random_state=random_state,
        test_size=test_size,
        val_size=val_size,
    )

    x_train, y_train = x_all.iloc[split.train_idx], y_all.iloc[split.train_idx]
    x_val, y_val = x_all.iloc[split.val_idx], y_all.iloc[split.val_idx]
    x_test, y_test = x_all.iloc[split.test_idx], y_all.iloc[split.test_idx]

    model = make_xgb_classifier(y_train, random_state=random_state, **(xgb_overrides or {}))
    model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], verbose=fit_verbose)

    proba_val = model.predict_proba(x_val)[:, 1]
    proba_test = model.predict_proba(x_test)[:, 1]
    selected_threshold = select_threshold_for_recall(y_val, proba_val, target_recall=target_recall)

    metrics = {
        "val": evaluate_binary_classifier(y_val, proba_val, selected_threshold),
        "test": evaluate_binary_classifier(y_test, proba_test, selected_threshold),
        "test_at_0_5": evaluate_binary_classifier(y_test, proba_test, 0.5),
    }

    predictions = make_predictions_dataframe(xgb_df, split, proba_val, proba_test, selected_threshold)
    importance = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    result = XGBoostSegmentResult(
        model=model,
        dataframe=xgb_df,
        feature_cols=feature_cols,
        split=split,
        selected_threshold=selected_threshold,
        metrics=metrics,
        predictions=predictions,
        importance=importance,
    )

    if model_output_path is not None:
        save_xgboost_model(result, model_output_path)

    return result


def save_xgboost_outputs(
    result: XGBoostSegmentResult,
    output_dir: str | Path,
    predictions_name: str = "xgboost_oriented_segments_predictions.csv",
    importance_name: str = "xgboost_feature_importance.csv",
) -> tuple[Path, Path]:
    """Save prediction and feature-importance CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / predictions_name
    importance_path = output_dir / importance_name
    result.predictions.to_csv(pred_path, index=False)
    result.importance.to_csv(importance_path, index=False)
    return pred_path, importance_path


def save_xgboost_model(
    result: XGBoostSegmentResult,
    path: str | Path,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Serialize the trained XGBoost bundle to one file for later pipeline use."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = XGBoostSegmentArtifact(
        model=result.model,
        feature_cols=list(result.feature_cols),
        threshold=float(result.selected_threshold),
        metadata={
            "split_seed": result.split.seed,
            "n_features": len(result.feature_cols),
            **(metadata or {}),
        },
    )
    with path.open("wb") as f:
        pickle.dump(artifact, f)
    return path


def load_xgboost_model(path: str | Path) -> XGBoostSegmentArtifact:
    """Load an XGBoost bundle saved with :func:`save_xgboost_model`."""
    path = Path(path)
    with path.open("rb") as f:
        artifact = pickle.load(f)

    if isinstance(artifact, XGBoostSegmentArtifact):
        return artifact

    # Backward-compatible fallback for dict artifacts.
    if isinstance(artifact, dict) and {"model", "feature_cols", "threshold"}.issubset(artifact):
        return XGBoostSegmentArtifact(
            model=artifact["model"],
            feature_cols=list(artifact["feature_cols"]),
            threshold=float(artifact["threshold"]),
            metadata=dict(artifact.get("metadata", {})),
        )

    raise TypeError(f"Unsupported XGBoost artifact format in {path}")


def prepare_segments_for_xgboost_prediction(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Apply the same feature engineering as in training and return columns in model order."""
    if df.empty:
        return pd.DataFrame(columns=list(feature_cols))

    xgb_df = add_neighbor_features(df)
    for col in feature_cols:
        if col not in xgb_df.columns:
            xgb_df[col] = 0.0

    x = xgb_df[list(feature_cols)].copy()
    numeric_cols = x.select_dtypes(include=[np.number]).columns.tolist()
    x[numeric_cols] = x[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return x


def predict_xgboost_on_segments(
    df: pd.DataFrame,
    artifact_or_path: XGBoostSegmentArtifact | str | Path,
) -> pd.DataFrame:
    """Return a copy of segment rows with XGBoost stenosis probability and label."""
    artifact = (
        load_xgboost_model(artifact_or_path)
        if isinstance(artifact_or_path, (str, Path))
        else artifact_or_path
    )
    out = df.copy()
    if out.empty:
        out["xgb_pred_proba"] = []
        out["xgb_pred_label"] = []
        return out

    x = prepare_segments_for_xgboost_prediction(out, artifact.feature_cols)
    proba = artifact.model.predict_proba(x)[:, 1]
    out["xgb_pred_proba"] = proba
    out["xgb_pred_label"] = (proba >= artifact.threshold).astype(int)
    out["xgb_threshold"] = float(artifact.threshold)
    return out
