from __future__ import annotations

from dataclasses import dataclass

import pandapower as pp

from violation_detector import ViolationReport


@dataclass(frozen=True)
class ScoreBreakdown:
    score: float
    remaining_violations: int
    overload_amount: float
    voltage_deviation: float
    intervention_cost: float


def score_network(report: ViolationReport, intervention_cost: float) -> ScoreBreakdown:
    if not report.converged:
        return ScoreBreakdown(1_000_000.0, 999, 999.0, 999.0, intervention_cost)

    overload = 0.0
    if not report.overloaded_lines.empty:
        overload += float(report.overloaded_lines["overload_percent"].clip(lower=0).sum())
    if not report.overloaded_trafos.empty:
        overload += float(report.overloaded_trafos["overload_percent"].clip(lower=0).sum())

    voltage = 0.0
    if not report.low_voltage_buses.empty:
        voltage += float(report.low_voltage_buses["low_deviation_pu"].clip(lower=0).sum())
    if not report.high_voltage_buses.empty:
        voltage += float(report.high_voltage_buses["high_deviation_pu"].clip(lower=0).sum())

    score = (
        report.violation_count * 10_000.0
        + overload * 100.0
        + voltage * 10_000.0
        + intervention_cost
    )
    return ScoreBreakdown(score, report.violation_count, overload, voltage, intervention_cost)


def estimate_action_cost(net: pp.pandapowerNet, action_type: str, magnitude: float) -> float:
    if action_type == "generator_redispatch":
        return abs(magnitude) * 1.0
    if action_type == "transformer_tap":
        return abs(magnitude) * 5.0
    if action_type == "voltage_setpoint":
        return abs(magnitude) * 100.0
    if action_type == "reactive_load_adjustment":
        return abs(magnitude) * 4.0
    if action_type == "line_switching":
        return 25.0
    if action_type == "load_curtailment":
        return abs(magnitude) * 50.0
    return abs(magnitude) * 10.0
