from datetime import datetime
from pathlib import Path
import json

import cv2
import numpy as np

from coronary_analysis.topology import classify_skeleton_pixels
from coronary_analysis.topology.junction_decision import JunctionDecision

from .analyze import AnalysisResult


TEMPLATE_PATH = Path(__file__).parent / "template.md"


def _to_bgr_image(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim == 2:
        return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _draw_xgboost_overlay(result: AnalysisResult) -> np.ndarray:
    """Return notebook-style visualization of oriented XGBoost segments.

    This mirrors the notebook visualization:
    - red: positive stenosis segment,
    - orange: negative segment,
    - when ground-truth labels are present: TP=red, TN=orange, FP=purple, FN=blue.

    The report does not usually have ARCADE annotations loaded, so the default
    inference view is original image + all oriented segment polygons colored by
    XGBoost prediction.
    """
    vis = _to_bgr_image(result.image)

    df = result.xgboost_segments
    if df is None or df.empty:
        return vis

    def _parse_vertices(value):
        if value is None:
            return None
        try:
            if isinstance(value, str):
                value = json.loads(value)
            vertices = np.asarray(value, dtype=np.float32)
        except Exception:
            return None
        if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] < 2:
            return None
        return np.round(vertices[:, :2]).astype(np.int32)

    def _row_int(row, name: str):
        if name not in row or row.get(name) is None:
            return None
        try:
            value = row.get(name)
            if isinstance(value, float) and np.isnan(value):
                return None
            return int(value)
        except Exception:
            return None

    # Draw negatives first and positives later, just like in the notebook where
    # the important/highlighted polygons remain visible on top.
    draw_df = df.copy()
    if "xgb_pred_label" in draw_df.columns:
        draw_df = draw_df.sort_values(["xgb_pred_label", "xgb_pred_proba"], ascending=[True, True])
    elif "label" in draw_df.columns:
        draw_df = draw_df.sort_values("label", ascending=True)

    for _, row in draw_df.iterrows():
        poly = _parse_vertices(row.get("oriented_box_vertices_xy"))
        if poly is None:
            continue

        true = _row_int(row, "label")
        pred = _row_int(row, "xgb_pred_label")

        # BGR colors chosen to match the notebook's named colors.
        if true is not None and pred is not None:
            if true == 1 and pred == 1:      # TP: red
                color, alpha, thickness = (0, 0, 255), 0.45, 2
            elif true == 0 and pred == 0:    # TN: orange
                color, alpha, thickness = (0, 165, 255), 0.16, 1
            elif true == 0 and pred == 1:    # FP: purple
                color, alpha, thickness = (128, 0, 128), 0.45, 2
            else:                            # FN: blue
                color, alpha, thickness = (255, 0, 0), 0.55, 2
        else:
            positive = pred == 1 if pred is not None else true == 1
            if positive:
                color, alpha, thickness = (0, 0, 255), 0.45, 2
            else:
                color, alpha, thickness = (0, 165, 255), 0.16, 1

        layer = vis.copy()
        cv2.fillPoly(layer, [poly], color)
        vis = cv2.addWeighted(layer, alpha, vis, 1.0 - alpha, 0)
        cv2.polylines(vis, [poly], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    return vis


def _save_visualizations(
    result: AnalysisResult,
    output_dir: Path,
    image_name: str,
) -> dict[str, str]:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_name).stem

    input_path = images_dir / f"{stem}_input.png"
    cv2.imwrite(str(input_path), _to_bgr_image(result.image))

    mask_path = images_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), result.mask.astype(np.uint8) * 255)

    graph_path = images_dir / f"{stem}_graph.png"
    base_rgb = result.image
    if base_rgb.ndim == 2:
        base_rgb = np.stack([base_rgb] * 3, axis=-1)
    vis = base_rgb.astype(np.float32) / 255.0 * 0.4
    vis[result.mask.astype(bool)] = [0.2, 0.3, 0.2]
    vis[result.skeleton.astype(bool)] = [1.0, 1.0, 1.0]
    vis = (vis * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(graph_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    xgboost_overlay_path = images_dir / f"{stem}_xgboost_overlay.png"
    cv2.imwrite(str(xgboost_overlay_path), _draw_xgboost_overlay(result))

    return {
        "input_image_path": f"images/{stem}_input.png",
        "mask_image_path": f"images/{stem}_mask.png",
        "graph_image_path": f"images/{stem}_graph.png",
        "xgboost_overlay_image_path": f"images/{stem}_xgboost_overlay.png",
    }


def _format_confidence(decision: JunctionDecision) -> str:
    confidence = getattr(decision, "confidence", None)
    if confidence is None:
        return "-"
    try:
        return f"{float(confidence):.3f}"
    except (TypeError, ValueError):
        return str(confidence)


def _format_junction_rows(junction_results: list[JunctionDecision]) -> str:
    rows = []

    for i, decision in enumerate(junction_results):
        center_y, center_x = decision.center

        rows.append(
            f"| {i} | "
            f"({float(center_y):.1f}, {float(center_x):.1f}) | "
            f"{decision.label.value} | "
            f"{_format_confidence(decision)} |"
        )

    if not rows:
        return "| - | - | - | - |"

    return "\n".join(rows)


def _format_xgboost_section(result: AnalysisResult) -> str:
    df = result.xgboost_segments
    stats = result.stats
    if df is None or "xgb_pred_proba" not in df.columns:
        return ""

    total = int(stats.get("xgboost_segments_total", len(df)))
    positive = int(stats.get("xgboost_segments_positive", 0))
    max_proba = stats.get("xgboost_max_probability")
    threshold = stats.get("xgboost_threshold")
    max_proba_txt = "-" if max_proba is None else f"{float(max_proba):.4f}"
    threshold_txt = "-" if threshold is None else f"{float(threshold):.4f}"

    top = df.sort_values("xgb_pred_proba", ascending=False).head(10)
    rows = []
    for _, row in top.iterrows():
        rows.append(
            f"| {int(row['branch_id'])} | {int(row['local_segment_id'])} | "
            f"{float(row['center_x']):.1f} | {float(row['center_y']):.1f} | "
            f"{float(row['xgb_pred_proba']):.4f} | {int(row['xgb_pred_label'])} |"
        )
    table_rows = "\n".join(rows) if rows else "| - | - | - | - | - | - |"

    return (
        "## XGBoost — predykcja stenozy na fragmentach\n\n"
        "Poniższa wizualizacja pokazuje maskę, szkielet oraz fragmenty wskazane przez XGBoosta. "
        "Czerwone prostokąty oznaczają fragmenty dodatnie, a liczba przy fragmencie to P(stenosis).\n\n"
        "![XGBoost overlay]({xgboost_overlay_image_path})\n\n"
        "| Metryka | Wartość |\n"
        "|---|---:|\n"
        f"| Liczba fragmentów | {total} |\n"
        f"| Fragmenty dodatnie | {positive} |\n"
        f"| Maksymalne prawdopodobieństwo | {max_proba_txt} |\n"
        f"| Threshold | {threshold_txt} |\n\n"
        "Top fragmenty według prawdopodobieństwa:\n\n"
        "| Branch ID | Segment ID | Center X | Center Y | P(stenosis) | Pred |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
        f"{table_rows}\n"
    )


def generate_report(
    result: AnalysisResult,
    image_path: str | Path,
    output_path: str | Path | None = None,
) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    stats = result.stats
    vessel_coverage = (result.mask > 0).sum() / result.mask.size * 100

    if output_path is not None:
        output_dir = Path(output_path).parent
        image_paths = _save_visualizations(result, output_dir, Path(image_path).name)
    else:
        image_paths = {
            "input_image_path": "(nie zapisano)",
            "mask_image_path": "(nie zapisano)",
            "graph_image_path": "(nie zapisano)",
            "xgboost_overlay_image_path": "(nie zapisano)",
        }

    branch_rows = "\n".join(
        f"| {b['branch_id']} | {b['length']:.1f} | {b['mean_diameter']:.1f} | "
        f"{b['min_diameter']:.1f} | {b['max_diameter']:.1f} |"
        for b in result.branch_details
    )

    endpoints, _ = classify_skeleton_pixels(result.skeleton)
    junction_rows = _format_junction_rows(result.junction_results)

    xgboost_section = _format_xgboost_section(result).format(**image_paths)

    report = template.format(
        image_name=Path(image_path).name,
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        width=result.image.shape[1],
        height=result.image.shape[0],
        total_vessel_length=f"{stats['total_vessel_length']:.1f}",
        num_branches=stats["num_branches"],
        longest_branch=f"{stats['longest_branch']:.1f}",
        shortest_branch=f"{stats['shortest_branch']:.1f}",
        mean_tortuosity=f"{stats['mean_tortuosity']:.3f}",
        vessel_coverage=f"{vessel_coverage:.2f}",
        num_endpoints=len(endpoints),
        branch_rows=branch_rows,
        junction_rows=junction_rows,
        xgboost_section=xgboost_section,
        n_certain_junctions=result.junction_counts.get("certain", 0),
        n_false_junctions=result.junction_counts.get("false", 0),
        n_not_junctions=result.junction_counts.get("not", 0),
        **image_paths,
    )

    if output_path is not None:
        Path(output_path).write_text(report, encoding="utf-8")

    return report
