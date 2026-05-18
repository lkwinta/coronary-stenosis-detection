from datetime import datetime
from pathlib import Path
from .analyze import AnalysisResult
from coronary_analysis.topology import classify_skeleton_pixels

import cv2
import numpy as np


TEMPLATE_PATH = Path(__file__).parent / "template.md"


def _save_visualizations(
    result: AnalysisResult,
    output_dir: Path,
    image_name: str,
) -> dict[str, str]:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_name).stem

    input_path = images_dir / f"{stem}_input.png"
    cv2.imwrite(str(input_path), result.image)

    mask_path = images_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), result.mask * 255)

    graph_path = images_dir / f"{stem}_graph.png"
    vis = np.stack([result.image] * 3, axis=-1).astype(np.float32) / 255.0 * 0.4
    vis[result.mask.astype(bool)] = [0.2, 0.3, 0.2]
    vis[result.skeleton.astype(bool)] = [1.0, 1.0, 1.0]
    vis = (vis * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(graph_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    return {
        "input_image_path": f"images/{stem}_input.png",
        "mask_image_path": f"images/{stem}_mask.png",
        "graph_image_path": f"images/{stem}_graph.png",
    }


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
        }

    branch_rows = "\n".join(
        f"| {b['branch_id']} | {b['length']:.1f} | {b['mean_diameter']:.1f} | {b['min_diameter']:.1f} | {b['max_diameter']:.1f} |"
        for b in result.branch_details
    )

    endpoints, _ = classify_skeleton_pixels(result.skeleton)

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
        **image_paths,
    )
    if output_path is not None:
        Path(output_path).write_text(report, encoding="utf-8")
    return report
