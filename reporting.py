from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_reports(
    screened_candidates: pd.DataFrame,
    validated_top_cases: pd.DataFrame,
    best_n1_plan: pd.DataFrame,
    output_folder: Path,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    screened_candidates.to_csv(output_folder / "screened_candidates.csv", index=False)
    validated_top_cases.to_csv(output_folder / "validated_top_cases.csv", index=False)
    best_n1_plan.to_csv(output_folder / "best_n1_plan.csv", index=False)
    plot_results(screened_candidates, validated_top_cases, output_folder)


def plot_results(screened_candidates: pd.DataFrame, validated_top_cases: pd.DataFrame, output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    if not screened_candidates.empty:
        top = screened_candidates.sort_values("risk_score", ascending=False).head(20)
        plt.figure(figsize=(12, 6))
        plt.bar(top["contingency_id"].astype(str), top["risk_score"])
        plt.xticks(rotation=70, ha="right")
        plt.ylabel("Predicted risk score")
        plt.title("Top risky contingencies by screening risk")
        plt.tight_layout()
        plt.savefig(output_folder / "top_risky_contingencies.png")
        plt.close()

    if not validated_top_cases.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(validated_top_cases["contingency_id"].astype(str), validated_top_cases["max_line_loading_percent"])
        plt.xticks(rotation=70, ha="right")
        plt.ylabel("Max line loading (%)")
        plt.title("Validated max line loading")
        plt.tight_layout()
        plt.savefig(output_folder / "validated_max_line_loading.png")
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(validated_top_cases["contingency_id"].astype(str), validated_top_cases["min_voltage_pu"])
        plt.xticks(rotation=70, ha="right")
        plt.ylabel("Min voltage (pu)")
        plt.title("Validated minimum voltage by contingency")
        plt.tight_layout()
        plt.savefig(output_folder / "validated_min_voltage.png")
        plt.close()

        violations = validated_top_cases.copy()
        violations["violation_count"] = violations["overloaded_lines"].fillna(0) + violations["voltage_violations"].fillna(0)
        plt.figure(figsize=(12, 6))
        plt.plot(violations["timestamp"], violations["violation_count"], marker="o", linestyle="")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Violation count")
        plt.title("Number of validated violations over time")
        plt.tight_layout()
        plt.savefig(output_folder / "violations_over_time.png")
        plt.close()
