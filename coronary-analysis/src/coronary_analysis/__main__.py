from pathlib import Path
from datetime import datetime
from coronary_analysis.pipeline import run_analysis, generate_report

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="coronary_analysis")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model weights (.pth)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save report (default: prints to stdout)",
    )
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--closing-radius", type=float, default=2)
    parser.add_argument("--max-hole-size", type=int, default=50)
    parser.add_argument("--min-object-size", type=int, default=50)
    parser.add_argument("--min-branch-length", type=int, default=15)
    args = parser.parse_args()
    result = run_analysis(
        image_path=args.image,
        model_path=args.model,
        encoder_name=args.encoder,
        img_size=args.img_size,
        threshold=args.threshold,
        closing_radius=args.closing_radius,
        max_hole_size=args.max_hole_size,
        min_object_size=args.min_object_size,
        min_branch_length=args.min_branch_length,
    )

    if args.output is None:
        repo_root = Path(__file__).parent.parent.parent.parent
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = repo_root / "raports" / f"report_{now}.md"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_report(result, image_path=args.image, output_path=output_path)

    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
