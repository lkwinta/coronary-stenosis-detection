# Refactor XGBoost notebook logic

Proponowane pliki do wklejenia do `src/coronary_analysis`:

```text
src/coronary_analysis/
  models/
    xgboost_segments.py
  metrics/
    enrichment.py
  visualization/
    xgboost_segments.py
```

Minimalny kod, który zostaje w notebooku, jest w:

```text
notebooks/xgboost_minimal_cells.py
```

## Co powinno zostać w notebooku

Notebook powinien być warstwą eksperymentu: importy, ustawienie ścieżek/parametrów,
uruchomienie pipeline'u, `display(...)`, wykresy i krótkie komentarze z wynikiem.

## Co powinno wylecieć do biblioteki

Do `src/coronary_analysis` powinny trafić funkcje powtarzalne: feature engineering,
split po `image_id`, konfiguracja modelu, trening, dobór thresholda, metryki, zapis CSV
i funkcje wizualizacyjne.

# Sposób użycia:

```bash
python -m coronary_analysis \
  --train-xgboost-csv notebooks/processed/stenosis_oriented_segments/oriented_vessel_segments_train_ready.csv \
  --xgboost-output models/xgboost_segments.pkl \
  --xgboost-outputs-dir notebooks/processed/stenosis_oriented_segments
```

```bash
python -m coronary_analysis \
  --image notebooks/raw_datasets/arcade/arcade/stenosis/train/images/12.png \
  --model models/best_segmentation_model.pth \
  --xgboost-model models/xgboost_segments.pkl \
  --xgboost-segments-output outputs/xgboost_segments.csv \
  --output outputs/report.md
```