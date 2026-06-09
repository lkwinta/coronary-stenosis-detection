"""Ranking/enrichment metrics for stenosis segment scores."""

from __future__ import annotations

import numpy as np
import pandas as pd


def enrichment_factor_table(
    y_true,
    y_score,
    top_fracs=(0.01, 0.02, 0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Compute Enrichment Factor for top-scored fractions of samples.

    EF@X% = positive rate in top X% divided by positive rate in the full set.
    """
    df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true).astype(int),
            "score": np.asarray(y_score),
        }
    ).sort_values("score", ascending=False).reset_index(drop=True)

    n_total = len(df)
    total_positives = int(df["y_true"].sum())
    baseline_positive_rate = total_positives / n_total if n_total else np.nan

    rows = []
    for frac in top_fracs:
        n_top = max(1, int(np.ceil(frac * n_total)))
        top_df = df.iloc[:n_top]
        positives_in_top = int(top_df["y_true"].sum())
        positive_rate_top = positives_in_top / n_top
        enrichment_factor = (
            positive_rate_top / baseline_positive_rate
            if baseline_positive_rate and baseline_positive_rate > 0
            else np.nan
        )
        recall_at_top = positives_in_top / total_positives if total_positives > 0 else np.nan
        rows.append(
            {
                "top_percent": frac * 100,
                "n_top": n_top,
                "positives_in_top": positives_in_top,
                "positive_rate_top": positive_rate_top,
                "baseline_positive_rate": baseline_positive_rate,
                "enrichment_factor": enrichment_factor,
                "recall_at_top": recall_at_top,
            }
        )

    return pd.DataFrame(rows)
