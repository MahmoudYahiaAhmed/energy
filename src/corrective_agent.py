from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Callable

import pandapower as pp

from grid_loader import run_power_flow
from scoring import ScoreBreakdown, estimate_action_cost, score_network
from violation_detector import ViolationReport, detect_violations


@dataclass
class CandidateAction:
    action_type: str
    description: str
    magnitude: float
    apply: Callable[[pp.pandapowerNet], None]


@dataclass
class CandidateResult:
    action: CandidateAction
    converged: bool
    safe: bool
    score: ScoreBreakdown
    violations: ViolationReport
    net: pp.pandapowerNet


@dataclass
class GreedyStep:
    step_number: int
    start_score: ScoreBreakdown
    chosen: CandidateResult


@dataclass
class AgentResult:
    observation: str
    thought: str
    candidates: list[CandidateResult]
    chosen: CandidateResult | None
    final_net: pp.pandapowerNet
    path: list[GreedyStep]
    stop_reason: str


def run_corrective_agent(
    net: pp.pandapowerNet,
    report: ViolationReport,
    max_steps: int = 12,
) -> AgentResult:
    observation = _observe(report)
    thought = _think(report)
    candidates: list[CandidateResult] = []
    path: list[GreedyStep] = []
    if report.is_safe:
        return AgentResult(
            observation=observation,
            thought=thought,
            candidates=candidates,
            chosen=None,
            final_net=deepcopy(net),
            path=path,
            stop_reason="already_stable",
        )

    current_net = deepcopy(net)
    current_report = report
    chosen: CandidateResult | None = None
    stop_reason = "max_steps_reached"

    for step_number in range(1, max_steps + 1):
        start_score = score_network(current_report, 0.0)
        step_candidates = evaluate_candidate_actions(current_net, current_report)
        candidates.extend(step_candidates)
        if not step_candidates:
            stop_reason = "no_candidates"
            break

        next_choice = _choose_best_candidate(step_candidates)
        if next_choice is None:
            stop_reason = "no_converged_candidate"
            break

        chosen = next_choice
        path.append(
            GreedyStep(
                step_number=step_number,
                start_score=start_score,
                chosen=next_choice,
            )
        )
        current_net = deepcopy(next_choice.net)
        current_report = next_choice.violations
        if current_report.is_safe:
            stop_reason = "stable"
            break

    final_net = deepcopy(chosen.net) if chosen else deepcopy(net)
    return AgentResult(
        observation=observation,
        thought=thought,
        candidates=candidates,
        chosen=chosen,
        final_net=final_net,
        path=path,
        stop_reason=stop_reason,
    )


def evaluate_candidate_actions(net: pp.pandapowerNet, report: ViolationReport) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    for action in generate_candidate_actions(net, report):
        trial = deepcopy(net)
        try:
            action.apply(trial)
            converged = run_power_flow(trial)
            trial_report = detect_violations(trial)
        except Exception:
            converged = False
            trial_report = ViolationReport(converged=False)

        cost = estimate_action_cost(trial, action.action_type, action.magnitude)
        score = score_network(trial_report, cost)
        candidates.append(
            CandidateResult(
                action=action,
                converged=converged,
                safe=trial_report.is_safe,
                score=score,
                violations=trial_report,
                net=trial,
            )
        )
    return candidates


def _choose_best_candidate(candidates: list[CandidateResult]) -> CandidateResult | None:
    safe_candidates = [candidate for candidate in candidates if candidate.safe]
    ranked_pool = safe_candidates or [candidate for candidate in candidates if candidate.converged]
    return min(ranked_pool, key=lambda candidate: candidate.score.score) if ranked_pool else None


def generate_candidate_actions(net: pp.pandapowerNet, report: ViolationReport) -> list[CandidateAction]:
    if report.is_safe:
        return []

    actions: list[CandidateAction] = []
    actions.extend(_generator_redispatch_actions(net, report))
    actions.extend(_voltage_setpoint_actions(net, report))
    actions.extend(_tap_actions(net, report))
    actions.extend(_line_switching_actions(net, report))
    actions.extend(_reactive_load_actions(net, report))
    actions.extend(_load_curtailment_actions(net, report))
    return actions


def _generator_redispatch_actions(net: pp.pandapowerNet, report: ViolationReport) -> list[CandidateAction]:
    actions: list[CandidateAction] = []
    if not len(net.gen):
        return actions

    stressed_buses = _stressed_buses(report)
    gen_rows = _in_service_rows(net.gen)
    if "controllable" in gen_rows.columns:
        gen_rows = gen_rows[gen_rows["controllable"].astype(bool)]

    for idx, row in gen_rows.iterrows():
        bus = int(row.bus)
        max_p = float(row.get("max_p_mw", row.p_mw * 1.3 + 1.0))
        min_p = float(row.get("min_p_mw", 0.0))
        current = float(row.p_mw)
        if bus in stressed_buses and current < max_p:
            delta = min(max_p - current, max(1.0, abs(current) * 0.1))
        elif current > min_p:
            delta = -min(current - min_p, max(1.0, abs(current) * 0.1))
        else:
            continue

        def apply(trial: pp.pandapowerNet, gen_idx=int(idx), change=float(delta)) -> None:
            trial.gen.at[gen_idx, "p_mw"] = float(trial.gen.at[gen_idx, "p_mw"]) + change

        direction = "increase" if delta > 0 else "decrease"
        actions.append(
            CandidateAction(
                "generator_redispatch",
                f"{direction.capitalize()} generator {idx} active power by {abs(delta):.2f} MW",
                delta,
                apply,
            )
        )
    return actions[:6]


def _tap_actions(net: pp.pandapowerNet, report: ViolationReport) -> list[CandidateAction]:
    actions: list[CandidateAction] = []
    if not len(net.trafo) or not (len(report.low_voltage_buses) or len(report.high_voltage_buses)):
        return actions

    low_voltage = len(report.low_voltage_buses) >= len(report.high_voltage_buses)
    for idx, row in net.trafo.iterrows():
        if not bool(row.get("in_service", True)) or "tap_pos" not in net.trafo.columns:
            continue
        tap_pos = row.get("tap_pos")
        if not _is_finite_number(tap_pos):
            continue
        tap_min = row.get("tap_min", tap_pos - 2)
        tap_max = row.get("tap_max", tap_pos + 2)
        if not _is_finite_number(tap_min):
            tap_min = tap_pos - 2
        if not _is_finite_number(tap_max):
            tap_max = tap_pos + 2
        step = -1 if low_voltage else 1
        new_tap = int(tap_pos + step)
        if new_tap < int(tap_min) or new_tap > int(tap_max):
            continue

        def apply(trial: pp.pandapowerNet, trafo_idx=int(idx), value=int(new_tap)) -> None:
            trial.trafo.at[trafo_idx, "tap_pos"] = value

        direction = "raise low-side voltage" if low_voltage else "reduce high voltage"
        actions.append(
            CandidateAction(
                "transformer_tap",
                f"Move transformer {idx} tap to {new_tap} to {direction}",
                1.0,
                apply,
            )
        )
    return actions[:4]


def _voltage_setpoint_actions(net: pp.pandapowerNet, report: ViolationReport) -> list[CandidateAction]:
    actions: list[CandidateAction] = []
    if not (len(report.low_voltage_buses) or len(report.high_voltage_buses)):
        return actions

    raise_voltage = len(report.low_voltage_buses) >= len(report.high_voltage_buses)
    direction = 1.0 if raise_voltage else -1.0
    verb = "Raise" if raise_voltage else "Lower"
    target_buses = _stressed_buses(report)

    if len(net.gen) and "vm_pu" in net.gen.columns:
        gen_rows = _in_service_rows(net.gen)
        if target_buses:
            local = gen_rows[gen_rows["bus"].astype(int).isin(target_buses)]
            gen_rows = local if len(local) else gen_rows
        for idx, row in gen_rows.head(6).iterrows():
            current = row.get("vm_pu", None)
            if not _is_finite_number(current):
                continue
            new_vm = _bounded_voltage_setpoint(float(current) + direction * 0.02)
            if math.isclose(new_vm, float(current), abs_tol=1e-6):
                continue

            def apply(trial: pp.pandapowerNet, gen_idx=int(idx), value=float(new_vm)) -> None:
                trial.gen.at[gen_idx, "vm_pu"] = value

            actions.append(
                CandidateAction(
                    "voltage_setpoint",
                    f"{verb} generator {idx} voltage setpoint to {new_vm:.3f} pu",
                    abs(new_vm - float(current)),
                    apply,
                )
            )

    if len(net.ext_grid) and "vm_pu" in net.ext_grid.columns:
        for idx, row in _in_service_rows(net.ext_grid).head(3).iterrows():
            current = row.get("vm_pu", None)
            if not _is_finite_number(current):
                continue
            new_vm = _bounded_voltage_setpoint(float(current) + direction * 0.02)
            if math.isclose(new_vm, float(current), abs_tol=1e-6):
                continue

            def apply(trial: pp.pandapowerNet, ext_idx=int(idx), value=float(new_vm)) -> None:
                trial.ext_grid.at[ext_idx, "vm_pu"] = value

            actions.append(
                CandidateAction(
                    "voltage_setpoint",
                    f"{verb} external grid {idx} voltage setpoint to {new_vm:.3f} pu",
                    abs(new_vm - float(current)),
                    apply,
                )
            )

    return actions[:8]


def _line_switching_actions(net: pp.pandapowerNet, report: ViolationReport) -> list[CandidateAction]:
    actions: list[CandidateAction] = []
    if report.overloaded_lines.empty:
        return actions
    overloaded = set(int(idx) for idx in report.overloaded_lines.index)
    for idx, row in net.line.iterrows():
        if int(idx) in overloaded or not bool(row.get("in_service", True)):
            continue

        def apply(trial: pp.pandapowerNet, line_idx=int(idx)) -> None:
            trial.line.at[line_idx, "in_service"] = False

        actions.append(CandidateAction("line_switching", f"Open non-overloaded parallel path candidate line {idx}", 1.0, apply))
        if len(actions) >= 3:
            break
    return actions


def _reactive_load_actions(net: pp.pandapowerNet, report: ViolationReport) -> list[CandidateAction]:
    actions: list[CandidateAction] = []
    if not len(net.load) or not (len(report.low_voltage_buses) or len(report.high_voltage_buses)):
        return actions

    high_voltage = len(report.high_voltage_buses) > len(report.low_voltage_buses)
    load_rows = _in_service_rows(net.load)
    stressed_buses = _stressed_buses(report)
    if stressed_buses:
        targeted = load_rows[load_rows["bus"].astype(int).isin(stressed_buses)]
        load_rows = targeted if len(targeted) else load_rows

    for idx, row in load_rows.sort_values("p_mw", ascending=False).head(6).iterrows():
        p_mw = abs(float(row.get("p_mw", 0.0)))
        current_q = float(row.get("q_mvar", 0.0))
        change = max(0.5, p_mw * 0.05)
        delta_q = change if high_voltage else -min(change, max(0.0, current_q))
        if math.isclose(delta_q, 0.0, abs_tol=1e-9):
            continue

        def apply(trial: pp.pandapowerNet, load_idx=int(idx), change_q=float(delta_q)) -> None:
            current = float(trial.load.at[load_idx, "q_mvar"]) if "q_mvar" in trial.load.columns else 0.0
            trial.load.at[load_idx, "q_mvar"] = current + change_q

        direction = "increase" if delta_q > 0 else "decrease"
        actions.append(
            CandidateAction(
                "reactive_load_adjustment",
                f"{direction.capitalize()} load {idx} reactive demand by {abs(delta_q):.2f} MVAr",
                delta_q,
                apply,
            )
        )

    return actions


def _load_curtailment_actions(net: pp.pandapowerNet, report: ViolationReport) -> list[CandidateAction]:
    actions: list[CandidateAction] = []
    if not len(net.load):
        return actions

    stressed_buses = _stressed_buses(report)
    load_rows = _in_service_rows(net.load)
    if stressed_buses:
        targeted = load_rows[load_rows["bus"].astype(int).isin(stressed_buses)]
        load_rows = targeted if len(targeted) else load_rows

    for fraction in (0.05, 0.10, 0.15):
        for idx, row in load_rows.sort_values("p_mw", ascending=False).head(4).iterrows():
            shed = min(float(row.p_mw) * fraction, float(row.p_mw))
            if shed <= 0:
                continue

            def apply(trial: pp.pandapowerNet, load_idx=int(idx), shed_mw=float(shed)) -> None:
                trial.load.at[load_idx, "p_mw"] = max(0.0, float(trial.load.at[load_idx, "p_mw"]) - shed_mw)

            actions.append(CandidateAction("load_curtailment", f"Curtail load {idx} by {shed:.2f} MW", shed, apply))
    return actions


def _stressed_buses(report: ViolationReport) -> set[int]:
    buses: set[int] = set()
    if not report.overloaded_lines.empty:
        buses.update(report.overloaded_lines["from_bus"].astype(int).tolist())
        buses.update(report.overloaded_lines["to_bus"].astype(int).tolist())
    if not report.low_voltage_buses.empty:
        buses.update(int(idx) for idx in report.low_voltage_buses.index)
    if not report.high_voltage_buses.empty:
        buses.update(int(idx) for idx in report.high_voltage_buses.index)
    return buses


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _in_service_rows(frame):
    if "in_service" not in frame.columns:
        return frame
    return frame[frame["in_service"].astype(bool)]


def _bounded_voltage_setpoint(value: float) -> float:
    return min(1.10, max(0.90, value))


def _observe(report: ViolationReport) -> str:
    if not report.converged:
        return "Power flow did not converge after the contingency."
    if report.is_safe:
        return "No thermal or voltage violations were detected."
    return (
        f"Detected {len(report.overloaded_lines)} overloaded lines, "
        f"{len(report.overloaded_trafos)} overloaded transformers, "
        f"{len(report.low_voltage_buses)} low-voltage buses, and "
        f"{len(report.high_voltage_buses)} high-voltage buses."
    )


def _think(report: ViolationReport) -> str:
    if not report.converged:
        return "Try conservative interventions that improve solvability, then validate with AC power flow."
    if len(report.overloaded_lines) or len(report.overloaded_trafos):
        return "Primary issue is thermal loading, so test redispatch, topology changes, and targeted load relief."
    if len(report.low_voltage_buses) or len(report.high_voltage_buses):
        return "Primary issue is voltage security, so test voltage setpoints, transformer taps, reactive load changes, and local generation changes."
    return "The contingency is already secure, so no corrective action is required."
