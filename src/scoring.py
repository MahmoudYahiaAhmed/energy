from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


def score_state(
    net: pp.pandapowerNet,
    report: ViolationReport,
    action: Any | None = None,
    config: Any | None = None,
) -> float:
    if not report.converged:
        return 1_000_000_000.0 + _action_penalty(action)

    line_excess_sq = 0.0
    if not report.overloaded_lines.empty:
        line_excess_sq = float(
            (report.overloaded_lines["overload_percent"].clip(lower=0.0) ** 2).sum()
        )

    trafo_excess_sq = 0.0
    if not report.overloaded_trafos.empty:
        trafo_excess_sq = float(
            (report.overloaded_trafos["overload_percent"].clip(lower=0.0) ** 2).sum()
        )

    voltage_dev_sq = 0.0
    if not report.low_voltage_buses.empty:
        voltage_dev_sq += float(
            (report.low_voltage_buses["low_deviation_pu"].clip(lower=0.0) ** 2).sum()
        )
    if not report.high_voltage_buses.empty:
        voltage_dev_sq += float(
            (report.high_voltage_buses["high_deviation_pu"].clip(lower=0.0) ** 2).sum()
        )

    return (
        1_000_000.0 * report.violation_count
        + 10_000.0 * line_excess_sq
        + 10_000.0 * trafo_excess_sq
        + 100_000.0 * voltage_dev_sq
        + _action_penalty(action)
    )


def _action_penalty(action: Any | None) -> float:
    if action is None:
        return 0.0
    cost = float(getattr(action, "cost", 0.0) or 0.0)
    disruptive_rank = float(getattr(action, "disruptive_rank", 0.0) or 0.0)
    return cost + 100.0 * disruptive_rank


def estimate_action_cost(net: pp.pandapowerNet, action_type: str, magnitude: float) -> float:
    if action_type in {"generator_redispatch", "gen_redispatch_pair"}:
        return abs(magnitude) * 1.0
    if action_type in {"transformer_tap", "trafo_tap_change"}:
        return abs(magnitude) * 5.0
    if action_type in {"voltage_setpoint", "gen_voltage_setpoint", "ext_grid_voltage_setpoint"}:
        return abs(magnitude) * 100.0
    if action_type == "reactive_load_adjustment":
        return abs(magnitude) * 4.0
    if action_type in {"line_switching", "line_switch"}:
        return 25.0
    if action_type == "load_curtailment":
        return abs(magnitude) * 50.0
    return abs(magnitude) * 10.0
