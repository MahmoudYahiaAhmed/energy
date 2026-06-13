from __future__ import annotations

import pandas as pd


def compute_risk_score(prediction: dict) -> float:
    score = 0.0
    if not bool(prediction.get("predicted_feasible", True)):
        score += 1_000_000.0

    max_loading = prediction.get("predicted_max_line_loading_percent")
    min_voltage = prediction.get("predicted_min_voltage_pu")
    max_voltage = prediction.get("predicted_max_voltage_pu")

    if max_loading is not None:
        score += max(0.0, float(max_loading) - 100.0) * 1_000.0
    if min_voltage is not None:
        score += max(0.0, 0.95 - float(min_voltage)) * 100_000.0
    if max_voltage is not None:
        score += max(0.0, float(max_voltage) - 1.05) * 100_000.0
    return score


def rank_candidates(prediction_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if prediction_df.empty:
        return prediction_df
    ranked = prediction_df.copy()
    if "risk_score" not in ranked.columns:
        ranked["risk_score"] = ranked.apply(lambda row: compute_risk_score(row.to_dict()), axis=1)
    return ranked.sort_values("risk_score", ascending=False).head(top_k).reset_index(drop=True)
