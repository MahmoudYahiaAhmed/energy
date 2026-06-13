from __future__ import annotations

import pandas as pd


def select_best_plan(validated_df: pd.DataFrame) -> pd.DataFrame:
    """Select the best current N-1 plan.

    Corrective actions are not implemented here yet. For now this ranks validated contingencies by
    severity and highlights secure cases first, then the least severe insecure cases.
    """
    if validated_df.empty:
        return validated_df
    df = validated_df.copy()
    df["severity_score"] = (
        (~df["converged"].astype(bool)).astype(float) * 1_000_000.0
        + df["overloaded_lines"].fillna(999).astype(float) * 10_000.0
        + df["voltage_violations"].fillna(999).astype(float) * 10_000.0
        + df["max_line_loading_percent"].fillna(1_000.0).clip(lower=100.0).sub(100.0) * 100.0
        + df["min_voltage_pu"].fillna(0.0).rsub(0.95).clip(lower=0.0) * 100_000.0
        + df["max_voltage_pu"].fillna(2.0).sub(1.05).clip(lower=0.0) * 100_000.0
    )
    return df.sort_values(["secure", "severity_score"], ascending=[False, True]).reset_index(drop=True)
