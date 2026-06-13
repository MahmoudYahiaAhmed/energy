from __future__ import annotations

import pandas as pd
import pandapower as pp

from risk_scoring import compute_risk_score, rank_candidates
from scenario_builder import BaseProfiles, apply_scenario_to_pandapower, capture_base_profiles


def dc_screen_contingencies(
    net: pp.pandapowerNet,
    scenarios: list[dict],
    contingencies: list[dict],
    top_k: int,
) -> pd.DataFrame:
    """Fast fallback screener for small grids.

    DC power flow is only used to rank risky cases. It is not the final engineering validator.
    """
    base_profiles = capture_base_profiles(net)
    rows: list[dict] = []
    for scenario in scenarios:
        apply_scenario_to_pandapower(net, scenario, base_profiles)
        for contingency in contingencies:
            original_status = _set_contingency_status(net, contingency, False)
            try:
                pp.rundcpp(net)
                converged = bool(getattr(net, "converged", False))
                max_loading = _max_line_loading(net)
                prediction = {
                    "predicted_feasible": converged,
                    "predicted_max_line_loading_percent": max_loading,
                    "predicted_min_voltage_pu": 1.0,
                    "predicted_max_voltage_pu": 1.0,
                    "predicted_branch_flows": None,
                }
            except Exception:
                prediction = {
                    "predicted_feasible": False,
                    "predicted_max_line_loading_percent": None,
                    "predicted_min_voltage_pu": None,
                    "predicted_max_voltage_pu": None,
                    "predicted_branch_flows": None,
                }
            finally:
                _restore_contingency_status(net, contingency, original_status)

            row = {
                "scenario_id": scenario["scenario_id"],
                "timestamp": scenario["timestamp"],
                "contingency_id": contingency["contingency_id"],
                "contingency_type": contingency["type"],
                "element_id": contingency["element_id"],
                **prediction,
            }
            row["risk_score"] = compute_risk_score(row)
            rows.append(row)

    return rank_candidates(pd.DataFrame(rows), top_k)


def _max_line_loading(net: pp.pandapowerNet) -> float | None:
    if hasattr(net, "res_line") and len(net.res_line) and "loading_percent" in net.res_line.columns:
        return float(net.res_line["loading_percent"].max())
    if hasattr(net, "res_line") and len(net.res_line) and "p_from_mw" in net.res_line.columns:
        return float(net.res_line["p_from_mw"].abs().max())
    return None


def _set_contingency_status(net: pp.pandapowerNet, contingency: dict, in_service: bool) -> bool:
    table = _table_for_type(net, contingency["type"])
    element_id = int(contingency["element_id"])
    original = bool(table.at[element_id, "in_service"]) if "in_service" in table.columns else True
    table.at[element_id, "in_service"] = in_service
    return original


def _restore_contingency_status(net: pp.pandapowerNet, contingency: dict, original_status: bool) -> None:
    table = _table_for_type(net, contingency["type"])
    table.at[int(contingency["element_id"]), "in_service"] = original_status


def _table_for_type(net: pp.pandapowerNet, contingency_type: str):
    if contingency_type == "line":
        return net.line
    if contingency_type == "trafo":
        return net.trafo
    if contingency_type == "gen":
        return net.gen
    raise ValueError(f"Unsupported contingency type: {contingency_type}")
