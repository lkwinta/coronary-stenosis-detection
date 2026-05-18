from datetime import datetime
from pathlib import Path
from .analyze import AnalysisResult


def generate_report(
    result: AnalysisResult,
    image_path: str | Path,
    output_path: str | Path | None = None,
) -> str:
    stats = result.stats
    mask = result.mask
    branch_type_names = {
        0: "endpoint-endpoint",
        1: "junction-endpoint",
        2: "junction-junction",
        3: "isolated cycle",
    }
    vessel_coverage = (mask > 0).sum() / mask.size * 100
    lines = [
        "# Coronary Vessel Analysis Report",
        "",
        f"**Source image:** {Path(image_path).name}  ",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Image dimensions:** {result.image.shape[1]} x {result.image.shape[0]}",
        "",
        "## Topology Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total vessel length | {stats['total_vessel_length']:.1f} px |",
        f"| Number of branches | {stats['num_branches']} |",
        f"| Longest branch | {stats['longest_branch']:.1f} px |",
        f"| Shortest branch | {stats['shortest_branch']:.1f} px |",
        f"| Mean tortuosity | {stats['mean_tortuosity']:.3f} |",
        f"| Vessel coverage | {vessel_coverage:.2f} % |",
        "",
        "## Branch Details",
        "",
        "| ID | Type | Length (px) | Mean Ø (px) | Min Ø (px) | Max Ø (px) |",
        "|---|---|---|---|---|---|",
    ]
    for b in result.branch_details:
        branch_type_str = branch_type_names.get(b["branch_type"], str(b["branch_type"]))
        lines.append(
            f"| {b['branch_id']} | {branch_type_str} | {b['length']:.1f} | {b['mean_diameter']:.1f} | {b['min_diameter']:.1f} | {b['max_diameter']:.1f} |"
        )
    lines += [
        "",
        "## Branch Type Distribution",
        "",
        "| Type | Count |",
        "|---|---|",
    ]
    for type_id, count in stats["branch_type_counts"].items():
        type_name = branch_type_names.get(int(type_id), str(type_id))
        lines.append(f"| {type_name} | {count} |")
    report_string = "\n".join(lines) + "\n"

    if output_path is not None:
        Path(output_path).write_text(report_string, encoding="utf-8")

    return report_string
