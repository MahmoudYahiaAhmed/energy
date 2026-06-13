from __future__ import annotations

import pandas as pd
import pandapower as pp

from scenario_builder import apply_scenario_to_pandapower, capture_base_profiles


def validate_top_cases(
    net: pp.pandapowerNet,
    scenarios: list[dict],
    top_cases: pd.DataFrame,
    min_voltage_pu: float = 0.95,
    max_voltage_pu: float = 1.05,
    max_line_loading_percent: float = 100.0,
    use_lightsim2grid: bool = False,
) -> pd.DataFrame:
    scenario_map = {scenario["scenario_id"]: scenario for scenario in scenarios}
    base_profiles = capture_base_profiles(net)
    rows: list[dict] = []

    for _, case in top_cases.iterrows():
        scenario = scenario_map[int(case["scenario_id"])]
        apply_scenario_to_pandapower(net, scenario, base_profiles)
        contingency = {
            "type": case["contingency_type"],
            "element_id": int(case["element_id"]),
            "contingency_id": case["contingency_id"],
        }
        original_status = _set_contingency_status(net, contingency, False)
        try:
            _run_ac_power_flow(net, use_lightsim2grid=use_lightsim2grid)
            converged = bool(getattr(net, "converged", False))
            row = _result_row(
                case=case,
                net=net,
                converged=converged,
                min_voltage_pu=min_voltage_pu,
                max_voltage_pu=max_voltage_pu,
                max_line_loading_percent=max_line_loading_percent,
                error="",
            )
        except Exception as exc:
            row = _result_row(
                case=case,
                net=net,
                converged=False,
                min_voltage_pu=min_voltage_pu,
                max_voltage_pu=max_voltage_pu,
                max_line_loading_percent=max_line_loading_percent,
                error=str(exc),
            )
        finally:
            _restore_contingency_status(net, contingency, original_status)
        rows.append(row)

    return pd.DataFrame(rows)


def _run_ac_power_flow(net: pp.pandapowerNet, use_lightsim2grid: bool) -> None:
    kwargs = {"numba": True, "init": "dc", "calculate_voltage_angles": True}
    if use_lightsim2grid:
        kwargs["lightsim2grid"] = True
    try:
        pp.runpp(net, **kwargs)
    except TypeError:
        kwargs.pop("lightsim2grid", None)
        pp.runpp(net, **kwargs)


def _result_row(
    case: pd.Series,
    net: pp.pandapowerNet,
    converged: bool,
    min_voltage_pu: float,
    max_voltage_pu: float,
    max_line_loading_percent: float,
    error: str,
) -> dict:
    if converged:
        min_vm = float(net.res_bus["vm_pu"].min())
        max_vm = float(net.res_bus["vm_pu"].max())
        max_loading = float(net.res_line["loading_percent"].max()) if len(net.res_line) else 0.0
        overloaded_lines = int((net.res_line["loading_percent"] > max_line_loading_percent).sum()) if len(net.res_line) else 0
        voltage_violations = int(((net.res_bus["vm_pu"] < min_voltage_pu) | (net.res_bus["vm_pu"] > max_voltage_pu)).sum())
        secure = overloaded_lines == 0 and voltage_violations == 0
    else:
        min_vm = None
        max_vm = None
        max_loading = None
        overloaded_lines = None
        voltage_violations = None
        secure = False

    return {
        "timestamp": case["timestamp"],
        "scenario_id": int(case["scenario_id"]),
        "contingency_id": case["contingency_id"],
        "contingency_type": case["contingency_type"],
        "element_id": int(case["element_id"]),
        "converged": converged,
        "min_voltage_pu": min_vm,
        "max_voltage_pu": max_vm,
        "max_line_loading_percent": max_loading,
        "overloaded_lines": overloaded_lines,
        "voltage_violations": voltage_violations,
        "secure": secure,
        "error": error,
    }


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
