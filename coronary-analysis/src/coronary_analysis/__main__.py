from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse

import pandas as pd

from coronary_analysis.models.xgboost_segments import (
    save_xgboost_outputs,
    train_xgboost_on_segments,
)
from coronary_analysis.pipeline import AnalysisConfig, generate_report, run_analysis


def _train_xgboost(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.train_xgboost_csv)
    result = train_xgboost_on_segments(
        df,
        random_state=args.xgboost_random_state,
        target_recall=args.xgboost_target_recall,
        model_output_path=args.xgboost_output,
        fit_verbose=args.xgboost_fit_verbose,
    )
    if args.xgboost_outputs_dir is not None:
        pred_path, importance_path = save_xgboost_outputs(result, args.xgboost_outputs_dir)
        print(f"XGBoost predictions saved to {pred_path}")
        print(f"XGBoost feature importance saved to {importance_path}")

    print(f"XGBoost model saved to {args.xgboost_output}")
    print(f"Selected threshold: {result.selected_threshold:.6f}")
    print("Validation metrics:")
    print(result.metrics["val"]["classification_report"])
    print("Test metrics:")
    print(result.metrics["test"]["classification_report"])


def _run_analysis(args: argparse.Namespace) -> None:
    if args.image is None or args.model is None:
        raise SystemExit("--image and --model are required for analysis mode")

    config = AnalysisConfig(
        encoder_name=args.encoder,
        img_size=args.img_size,
        threshold=args.threshold,
        closing_radius=args.closing_radius,
        max_hole_size=args.max_hole_size,
        min_object_size=args.min_object_size,
        min_branch_length=args.min_branch_length,
        xgboost_model_path=args.xgboost_model,
    )
    result = run_analysis(
        image_path=args.image,
        model_path=args.model,
        config=config,
    )

    if args.output is None:
        repo_root = Path(__file__).parent.parent.parent.parent
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = repo_root / "raports" / f"report_{now}.md"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_report(result, image_path=args.image, output_path=output_path)

    if args.xgboost_segments_output is not None and result.xgboost_segments is not None:
        xgb_path = Path(args.xgboost_segments_output)
        xgb_path.parent.mkdir(parents=True, exist_ok=True)
        result.xgboost_segments.to_csv(xgb_path, index=False)
        print(f"XGBoost segment results saved to {xgb_path}")

    print(f"Report saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="coronary_analysis")

    # Analysis mode
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--model", type=str, default=None, help="Path to segmentation model weights (.pth)")
    parser.add_argument("--output", type=str, default=None, help="Path to save report")
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--closing-radius", type=float, default=2)
    parser.add_argument("--max-hole-size", type=int, default=50)
    parser.add_argument("--min-object-size", type=int, default=50)
    parser.add_argument("--min-branch-length", type=int, default=15)
    parser.add_argument("--xgboost-model", type=str, default=None, help="Path to saved XGBoost .pkl model bundle")
    parser.add_argument("--xgboost-segments-output", type=str, default=None, help="Optional CSV path for segment-level XGBoost results")

    # Training mode
    parser.add_argument("--train-xgboost-csv", type=str, default=None, help="Train XGBoost from oriented segment CSV and exit")
    parser.add_argument("--xgboost-output", type=str, default="models/xgboost_segments.pkl", help="Where to save trained XGBoost bundle")
    parser.add_argument("--xgboost-outputs-dir", type=str, default=None, help="Optional dir for training predictions/importances")
    parser.add_argument("--xgboost-random-state", type=int, default=42)
    parser.add_argument("--xgboost-target-recall", type=float, default=0.85)
    parser.add_argument("--xgboost-fit-verbose", type=int, default=50)

    args = parser.parse_args()
    if args.train_xgboost_csv is not None:
        _train_xgboost(args)
    else:
        _run_analysis(args)


if __name__ == "__main__":
    main()
