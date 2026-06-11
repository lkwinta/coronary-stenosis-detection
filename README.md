# Coronary Stenosis Detection

This repository contains an experimental pipeline for coronary artery image analysis. It combines vessel segmentation, skeletonization, vessel graph extraction, junction classification, and branch-level measurements that can be used to inspect coronary stenosis candidates.

The reusable Python code lives in the local `coronary-analysis` package. The notebooks are used for dataset checks, model training, model loading, topology experiments, and XGBoost-based segment analysis.

## Setup

The project uses Python 3.12 and `uv`.

```bash
uv sync
```

This installs the root project and the editable local package from `coronary-analysis/`.

## Run the Pipeline

The main pipeline is exposed as a Python module:

```bash
python -m coronary_analysis \
  --image notebooks/raw_datasets/arcade/arcade/stenosis/train/images/12.png \
  --model models/best_segmentation_model.pth \
  --xgboost-model models/xgboost_segments.pkl \
  --xgboost-segments-output outputs/xgboost_segments.csv \
  --output outputs/report.md
```

Model training

```bash
python -m coronary_analysis \
  --train-xgboost-csv notebooks/processed/stenosis_oriented_segments/oriented_vessel_segments_train_ready.csv \
  --xgboost-output models/xgboost_segments.pkl \
  --xgboost-outputs-dir notebooks/processed/stenosis_oriented_segments
```

The command loads an angiography image, predicts the vessel mask, cleans and skeletonizes it, builds the vessel topology, classifies junctions, estimates branch diameters, and writes a Markdown report. If `--output` is omitted, a timestamped report is saved in `raports/`.

Useful options:

```bash
uv run python -m coronary_analysis --help
```

## Project Layout

- `coronary-analysis/src/coronary_analysis/` - reusable library code.
- `coronary-analysis/src/coronary_analysis/pipeline/` - end-to-end analysis and report generation.
- `coronary-analysis/src/coronary_analysis/datasets/` - dataset loaders for DCA1, FS-CAD, LM-CAD, and ARCADE syntax data.
- `coronary-analysis/src/coronary_analysis/models/` - segmentation and XGBoost model definitions.
- `coronary-analysis/src/coronary_analysis/topology/` - mask cleaning, skeletonization, graph extraction, branch metrics, and junction logic.
- `notebooks/` - exploratory and training notebooks.
- `notebooks/raw_datasets/` - extracted datasets used by the notebooks.
- `notebooks/downloads/` - downloaded dataset archives.
- `models/` - trained and pretrained model weights.
- `raports/` - project notes.

## Notebooks

The main notebooks are:

- `notebooks/datasets.ipynb` - dataset loading and examples.
- `notebooks/segmentation_pretrain.ipynb` - segmentation pretraining.
- `notebooks/segmentation_finetune.ipynb` - segmentation fine-tuning.
- `notebooks/load_model.ipynb` - loading and checking trained models.
- `notebooks/skeletonize.ipynb` - topology, skeletonization, and branch measurements.
- `notebooks/junction_decision.ipynb` - junction classification experiments.
- `notebooks/xgboost.ipynb` - XGBoost segment model experiments.
- `notebooks/xgboost_test.ipynb` - XGBoost model testing.

Start Jupyter with:

```bash
uv run jupyter lab notebooks
```