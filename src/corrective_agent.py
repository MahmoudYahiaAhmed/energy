from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass, field
import logging
import math
from typing import Any, Literal

import numpy as np
import pandas as pd
import pandapower as pp

from contingency import Contingency, apply_contingency
from grid_loader import PowerFlowMode, run_power_flow
from scoring import ScoreBreakdown, estimate_action_cost, score_network, score_state
from violation_detector import ViolationReport, detect_violations

logger = logging.getLogger(__name__)

PerformanceMode = Literal["balanced", "fast", "max_speed", "auto"]


@dataclass(frozen=True)
class ActionSpaceConfig:
    redispatch_step_fracs: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)
    redispatch_max_mw_per_action: float = 100.0
    voltage_setpoint_deltas_pu: tuple[float, ...] = (-0.02, -0.01, -0.005, 0.005, 0.01, 0.02)
    ext_grid_voltage_deltas_pu: tuple[float, ...] = (-0.02, -0.01, 0.01, 0.02)
    tap_step_deltas: tuple[int, ...] = (-2, -1, 1, 2)
    reactive_load_step_fracs: tuple[float, ...] = (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20)
    load_curtailment_fracs: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)
    max_total_curtailment_frac: float = 0.15
    local_bus_radius: int = 3
    candidate_dedup: bool = True
    line_switch_allowlist: tuple[int, ...] = ()
    allow_islanding: bool = False


@dataclass(frozen=True)
class CorrectiveOptimizerConfig:
    mode: Literal["ac", "dc"] = "ac"
    use_gridsfm: bool = False
    require_gridsfm: bool = False
    gridsfm_checkpoint: str | None = None
    gridsfm_device: str = "cpu"
    gridsfm_min_buses: int = 500
    gridsfm_feas_threshold: float = 0.80
    gridsfm_screen_top_k: int = 20
    max_candidates_total: int = 500
    max_candidates_per_type: int = 100
    max_gen_redispatch_candidates: int = 40
    max_gen_voltage_setpoint_candidates: int = 30
    max_trafo_tap_candidates: int = 30
    max_ext_grid_voltage_candidates: int = 10
    max_reactive_load_candidates: int = 30
    max_load_curtailment_candidates: int = 20
    max_line_switch_candidates: int = 10
    max_greedy_steps: int = 10
    require_solver_validation: bool = True
    include_voltage: bool = True
    allow_line_switching: bool = False
    allow_load_curtailment: bool = True
    allow_reactive_load_adjustment: bool = True
    allow_gen_redispatch: bool = True
    allow_gen_voltage_control: bool = True
    allow_ext_grid_voltage_control: bool = True
    allow_trafo_tap_control: bool = True
    random_seed: int = 42
    action_space: ActionSpaceConfig = field(default_factory=ActionSpaceConfig)


@dataclass(frozen=True)
class CorrectiveAction:
    action_type: str
    target_table: str
    target_index: int | tuple[int, ...]
    params: dict[str, Any]
    cost: float
    disruptive_rank: int
    reason: str

    @property
    def description(self) -> str:
        return self.reason

    @property
    def magnitude(self) -> float:
        for key in ("delta_mw", "delta_p_mw", "delta_q_mvar", "curtail_p_mw", "delta_pu", "tap_delta"):
            if key in self.params:
                return float(self.params[key])
        return float(self.cost)

    def signature(self) -> tuple:
        params = tuple(sorted((key, _hashable_value(value)) for key, value in self.params.items()))
        return (self.action_type, self.target_table, _hashable_value(self.target_index), params)

    def apply(self, net: pp.pandapowerNet) -> None:
        if self.action_type == "gen_redispatch_pair":
            up_idx, down_idx = self.target_index
            delta = float(self.params["delta_mw"])
            net.gen.at[int(up_idx), "p_mw"] = float(net.gen.at[int(up_idx), "p_mw"]) + delta
            net.gen.at[int(down_idx), "p_mw"] = float(net.gen.at[int(down_idx), "p_mw"]) - delta
        elif self.action_type == "gen_voltage_setpoint":
            net.gen.at[int(self.target_index), "vm_pu"] = float(self.params["vm_pu"])
        elif self.action_type == "ext_grid_voltage_setpoint":
            net.ext_grid.at[int(self.target_index), "vm_pu"] = float(self.params["vm_pu"])
        elif self.action_type == "trafo_tap_change":
            net.trafo.at[int(self.target_index), "tap_pos"] = int(self.params["tap_pos"])
        elif self.action_type == "line_switch":
            net.line.at[int(self.target_index), "in_service"] = bool(self.params["in_service"])
        elif self.action_type == "reactive_load_adjustment":
            net.load.at[int(self.target_index), "q_mvar"] = float(self.params["q_mvar"])
        elif self.action_type == "load_curtailment":
            net.load.at[int(self.target_index), "p_mw"] = float(self.params["p_mw"])
            if "q_mvar" in net.load.columns:
                net.load.at[int(self.target_index), "q_mvar"] = float(self.params["q_mvar"])
        else:
            raise ValueError(f"Unsupported corrective action type: {self.action_type}")


CandidateAction = CorrectiveAction


@dataclass
class CandidateResult:
    action: CorrectiveAction
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


@dataclass
class OptimizationResult:
    status: str
    actions: list[CorrectiveAction]
    initial_score: float
    final_score: float
    initial_violations: ViolationReport
    final_violations: ViolationReport
    gridsfm_used: bool
    gridsfm_skip_reason: str | None
    step_logs: list[dict[str, Any]]
    net: pp.pandapowerNet


def optimize_post_contingency(
    net: pp.pandapowerNet,
    contingency: Contingency | None,
    config: CorrectiveOptimizerConfig | None = None,
) -> OptimizationResult:
    config = config or CorrectiveOptimizerConfig()
    current = apply_contingency(net, contingency) if contingency is not None else deepcopy(net)

    run_power_flow(current, mode=config.mode)
    report = detect_violations(current, include_voltage=config.include_voltage and config.mode == "ac")
    initial_report = report
    current_score = score_state(current, report, config=config)
    initial_score = current_score
    history: list[CorrectiveAction] = []
    step_logs: list[dict[str, Any]] = []
    gridsfm_used = False
    gridsfm_skip_reason: str | None = None

    adapter = None
    if config.use_gridsfm:
        from gridsfm_adapter import GridSFMAdapter

        adapter = GridSFMAdapter(
            checkpoint_path=config.gridsfm_checkpoint,
            device=config.gridsfm_device,
            min_buses=config.gridsfm_min_buses,
            feasibility_threshold=config.gridsfm_feas_threshold,
        )
        supported, reason = adapter.is_supported_net(current)
        if not supported:
            gridsfm_skip_reason = reason
            logger.warning("GridSFM disabled: %s Using pandapower-only validation.", reason)
            if config.require_gridsfm:
                raise RuntimeError(f"GridSFM required but unavailable: {reason}")

    if report.is_safe:
        return OptimizationResult(
            status="already_safe",
            actions=[],
            initial_score=initial_score,
            final_score=current_score,
            initial_violations=initial_report,
            final_violations=report,
            gridsfm_used=False,
            gridsfm_skip_reason=gridsfm_skip_reason,
            step_logs=[],
            net=current,
        )

    for step in range(1, config.max_greedy_steps + 1):
        candidates = generate_corrective_candidates(current, report, config)
        step_log: dict[str, Any] = {
            "step": step,
            "start_score": current_score,
            "violations": report.to_summary(),
            "candidate_counts": dict(Counter(action.action_type for action in candidates)),
            "gridsfm_used": False,
            "validated": [],
        }
        if not candidates:
            step_logs.append(step_log)
            return _optimization_result(
                "no_candidates", history, initial_score, current_score, initial_report, report,
                gridsfm_used, gridsfm_skip_reason, step_logs, current
            )

        candidates_to_validate = candidates
        if config.use_gridsfm and adapter is not None and gridsfm_skip_reason is None:
            supported, reason = adapter.is_supported_net(current)
            if supported:
                ranked = adapter.rank_candidates(current, candidates, config)
                candidates_to_validate = [prediction.action for prediction in ranked[: config.gridsfm_screen_top_k]]
                gridsfm_used = True
                step_log["gridsfm_used"] = True
                step_log["gridsfm_top_k"] = [
                    {
                        "rank": idx + 1,
                        "action": prediction.action.description,
                        "predicted_score": prediction.predicted_score,
                        "feasibility": prediction.feasibility,
                        "failed": prediction.failed,
                    }
                    for idx, prediction in enumerate(ranked[: config.gridsfm_screen_top_k])
                ]
            else:
                gridsfm_skip_reason = reason
                logger.warning("GridSFM disabled: %s Using pandapower-only validation.", reason)
                if config.require_gridsfm:
                    raise RuntimeError(f"GridSFM required but unavailable: {reason}")
                step_log["gridsfm_skip_reason"] = reason

        best: tuple[CorrectiveAction, pp.pandapowerNet, ViolationReport, float] | None = None
        best_safe: tuple[CorrectiveAction, pp.pandapowerNet, ViolationReport, float] | None = None
        for action in candidates_to_validate:
            trial = deepcopy(current)
            try:
                action.apply(trial)
                pf_ok = run_power_flow(trial, mode=config.mode)
                trial_report = detect_violations(
                    trial, include_voltage=config.include_voltage and config.mode == "ac"
                )
            except Exception as exc:
                logger.debug("Candidate failed during AC validation: %s", exc)
                pf_ok = False
                trial_report = ViolationReport(converged=False)
            trial_score = score_state(trial, trial_report, action=action, config=config)
            step_log["validated"].append(
                {
                    "action": action.description,
                    "type": action.action_type,
                    "pf_ok": pf_ok,
                    "score": trial_score,
                    "violations": trial_report.to_summary(),
                }
            )
            if not pf_ok:
                continue
            choice = (action, trial, trial_report, trial_score)
            if trial_report.is_safe:
                if best_safe is None or _candidate_sort_key(choice) < _candidate_sort_key(best_safe):
                    best_safe = choice
            if trial_score < current_score - 1e-6:
                if best is None or trial_score < best[3]:
                    best = choice

        accepted = best_safe or best
        if accepted is None and config.use_gridsfm and len(candidates_to_validate) < len(candidates):
            fallback_pool = [action for action in candidates if action not in candidates_to_validate]
            for action in fallback_pool[: config.gridsfm_screen_top_k]:
                trial = deepcopy(current)
                try:
                    action.apply(trial)
                    pf_ok = run_power_flow(trial, mode=config.mode)
                    trial_report = detect_violations(
                        trial, include_voltage=config.include_voltage and config.mode == "ac"
                    )
                except Exception:
                    pf_ok = False
                    trial_report = ViolationReport(converged=False)
                trial_score = score_state(trial, trial_report, action=action, config=config)
                step_log["validated"].append(
                    {
                        "action": action.description,
                        "type": action.action_type,
                        "pf_ok": pf_ok,
                        "score": trial_score,
                        "fallback_validation": True,
                    }
                )
                if pf_ok and trial_score < current_score - 1e-6:
                    accepted = (action, trial, trial_report, trial_score)
                    break

        if accepted is None:
            step_logs.append(step_log)
            return _optimization_result(
                "no_improving_solver_validated_action", history, initial_score, current_score,
                initial_report, report, gridsfm_used, gridsfm_skip_reason, step_logs, current
            )

        action, current, report, current_score = accepted
        history.append(action)
        step_log["accepted_action"] = action.description
        step_log["end_score"] = current_score
        step_log["remaining_violations"] = report.to_summary()
        step_logs.append(step_log)
        logger.info("Accepted corrective action at step %s: %s", step, action.description)
        if report.is_safe:
            return _optimization_result(
                "safe", history, initial_score, current_score, initial_report, report,
                gridsfm_used, gridsfm_skip_reason, step_logs, current
            )

    return _optimization_result(
        "max_steps_reached", history, initial_score, current_score, initial_report, report,
        gridsfm_used, gridsfm_skip_reason, step_logs, current
    )


def run_corrective_agent(
    net: pp.pandapowerNet,
    report: ViolationReport,
    max_steps: int = 12,
    power_flow_mode: PowerFlowMode = "dc",
    performance_mode: PerformanceMode = "auto",
) -> AgentResult:
    observation = _observe(report, power_flow_mode)
    thought = _think(report, power_flow_mode)
    config = _agent_optimizer_config(
        net=net,
        power_flow_mode=power_flow_mode,
        max_steps=max_steps,
        performance_mode=performance_mode,
    )
    if report.is_safe:
        return AgentResult(observation, thought, [], None, deepcopy(net), [], "already_stable")

    current_net = deepcopy(net)
    current_report = report
    candidates: list[CandidateResult] = []
    path: list[GreedyStep] = []
    chosen: CandidateResult | None = None
    stop_reason = "max_steps_reached"
    seen_states = {_state_signature(current_net)}

    for step_number in range(1, max_steps + 1):
        start_score = score_network(current_report, 0.0)
        step_candidates = evaluate_candidate_actions(current_net, current_report, power_flow_mode, config)
        candidates.extend(step_candidates)
        if not step_candidates:
            stop_reason = "no_candidates"
            break

        next_choice = _choose_best_candidate(step_candidates, start_score, seen_states)
        if next_choice is None:
            stop_reason = "no_improving_candidate"
            break

        next_signature = _state_signature(next_choice.net)
        if next_signature in seen_states:
            stop_reason = "no_improving_candidate"
            break

        chosen = next_choice
        path.append(GreedyStep(step_number, start_score, next_choice))
        current_net = deepcopy(next_choice.net)
        current_report = next_choice.violations
        seen_states.add(next_signature)
        if current_report.is_safe:
            stop_reason = "stable"
            break

    final_net = deepcopy(chosen.net) if chosen else deepcopy(net)
    return AgentResult(observation, thought, candidates, chosen, final_net, path, stop_reason)


def evaluate_candidate_actions(
    net: pp.pandapowerNet,
    report: ViolationReport,
    power_flow_mode: PowerFlowMode = "dc",
    config: CorrectiveOptimizerConfig | None = None,
) -> list[CandidateResult]:
    config = config or CorrectiveOptimizerConfig(
        mode=power_flow_mode,
        include_voltage=power_flow_mode == "ac",
        allow_line_switching=False,
        use_gridsfm=False,
    )
    candidates: list[CandidateResult] = []
    for action in generate_corrective_candidates(net, report, config):
        trial = deepcopy(net)
        try:
            action.apply(trial)
            converged = run_power_flow(trial, mode=power_flow_mode)
            trial_report = detect_violations(trial, include_voltage=power_flow_mode == "ac")
        except Exception:
            converged = False
            trial_report = ViolationReport(converged=False)
        cost = estimate_action_cost(trial, action.action_type, action.magnitude)
        score = score_network(trial_report, cost)
        candidates.append(CandidateResult(action, converged, trial_report.is_safe, score, trial_report, trial))
    return candidates


def generate_candidate_actions(
    net: pp.pandapowerNet,
    report: ViolationReport,
    power_flow_mode: PowerFlowMode = "dc",
) -> list[CorrectiveAction]:
    config = CorrectiveOptimizerConfig(
        mode=power_flow_mode,
        include_voltage=power_flow_mode == "ac",
        allow_line_switching=False,
        use_gridsfm=False,
    )
    return generate_corrective_candidates(net, report, config)


def _agent_optimizer_config(
    net: pp.pandapowerNet,
    power_flow_mode: PowerFlowMode,
    max_steps: int,
    performance_mode: PerformanceMode,
) -> CorrectiveOptimizerConfig:
    bus_count = int(len(net.bus)) if hasattr(net, "bus") else 0
    profile = performance_mode
    if performance_mode == "auto":
        profile = "max_speed" if bus_count >= 1000 else "fast" if bus_count >= 118 else "balanced"

    base_kwargs: dict[str, Any] = {
        "mode": power_flow_mode,
        "max_greedy_steps": max_steps,
        "include_voltage": power_flow_mode == "ac",
        "allow_line_switching": False,
        "use_gridsfm": False,
    }

    if profile == "balanced":
        return CorrectiveOptimizerConfig(**base_kwargs)

    if profile == "fast":
        return CorrectiveOptimizerConfig(
            **base_kwargs,
            max_greedy_steps=min(max_steps, 8),
            max_candidates_total=180,
            max_candidates_per_type=45,
            max_gen_redispatch_candidates=24,
            max_gen_voltage_setpoint_candidates=20,
            max_trafo_tap_candidates=16,
            max_ext_grid_voltage_candidates=8,
            max_reactive_load_candidates=10,
            max_load_curtailment_candidates=10,
            action_space=ActionSpaceConfig(
                redispatch_step_fracs=(0.05, 0.10),
                redispatch_max_mw_per_action=80.0,
                voltage_setpoint_deltas_pu=(-0.01, 0.01),
                ext_grid_voltage_deltas_pu=(-0.01, 0.01),
                tap_step_deltas=(-1, 1),
                reactive_load_step_fracs=(-0.10, 0.10),
                load_curtailment_fracs=(0.02, 0.05),
                max_total_curtailment_frac=0.10,
                local_bus_radius=2,
                candidate_dedup=True,
                line_switch_allowlist=(),
                allow_islanding=False,
            ),
        )

    return CorrectiveOptimizerConfig(
        **base_kwargs,
        max_greedy_steps=min(max_steps, 4),
        max_candidates_total=72,
        max_candidates_per_type=18,
        max_gen_redispatch_candidates=12,
        max_gen_voltage_setpoint_candidates=8,
        max_trafo_tap_candidates=8,
        max_ext_grid_voltage_candidates=4,
        max_reactive_load_candidates=0,
        max_load_curtailment_candidates=8,
        allow_reactive_load_adjustment=False,
        action_space=ActionSpaceConfig(
            redispatch_step_fracs=(0.10,),
            redispatch_max_mw_per_action=60.0,
            voltage_setpoint_deltas_pu=(-0.01, 0.01),
            ext_grid_voltage_deltas_pu=(-0.01, 0.01),
            tap_step_deltas=(-1, 1),
            reactive_load_step_fracs=(),
            load_curtailment_fracs=(0.05,),
            max_total_curtailment_frac=0.08,
            local_bus_radius=1,
            candidate_dedup=True,
            line_switch_allowlist=(),
            allow_islanding=False,
        ),
    )


def generate_corrective_candidates(
    net: pp.pandapowerNet,
    report: ViolationReport,
    config: CorrectiveOptimizerConfig | ActionSpaceConfig | None = None,
) -> list[CorrectiveAction]:
    if report.is_safe:
        return []
    if config is None:
        optimizer_config = CorrectiveOptimizerConfig()
    elif isinstance(config, ActionSpaceConfig):
        optimizer_config = CorrectiveOptimizerConfig(action_space=config)
    else:
        optimizer_config = config

    graph = _build_bus_graph(net)
    affected = _affected_buses(net, report, graph, optimizer_config.action_space.local_bus_radius)
    actions: list[CorrectiveAction] = []
    if optimizer_config.allow_gen_redispatch:
        actions.extend(_generator_redispatch_actions(net, report, affected, graph, optimizer_config))
    if optimizer_config.mode == "ac" and optimizer_config.allow_gen_voltage_control:
        actions.extend(_gen_voltage_setpoint_actions(net, affected, optimizer_config))
    if optimizer_config.mode == "ac" and optimizer_config.allow_ext_grid_voltage_control:
        actions.extend(_ext_grid_voltage_setpoint_actions(net, affected, optimizer_config))
    if optimizer_config.mode == "ac" and optimizer_config.allow_trafo_tap_control:
        actions.extend(_tap_actions(net, report, affected, optimizer_config))
    if optimizer_config.allow_line_switching:
        actions.extend(_line_switching_actions(net, report, graph, optimizer_config))
    if optimizer_config.mode == "ac" and optimizer_config.allow_reactive_load_adjustment:
        actions.extend(_reactive_load_actions(net, report, affected, optimizer_config))
    if optimizer_config.allow_load_curtailment:
        actions.extend(_load_curtailment_actions(net, affected, optimizer_config))
    return _deduplicate_sort_and_cap(actions, optimizer_config, affected, graph)


def _generator_redispatch_actions(
    net: pp.pandapowerNet,
    report: ViolationReport,
    affected: set[int],
    graph: dict[int, set[int]],
    config: CorrectiveOptimizerConfig,
) -> list[CorrectiveAction]:
    if not len(net.gen):
        return []
    rows = _in_service_rows(net.gen)
    if "controllable" in rows.columns:
        rows = rows[rows["controllable"].fillna(True).astype(bool)]
    if rows.empty:
        return []

    local = rows[rows["bus"].astype(int).isin(affected)] if affected else rows.iloc[0:0]
    nonlocal_rows = rows.drop(local.index, errors="ignore")
    pairs: list[tuple[int, int]] = []
    for up_rows, down_rows in ((local, nonlocal_rows), (nonlocal_rows, local), (local, local), (rows, rows)):
        for up_idx in up_rows.index:
            for down_idx in down_rows.index:
                if int(up_idx) != int(down_idx):
                    pairs.append((int(up_idx), int(down_idx)))
        if pairs:
            break

    actions: list[CorrectiveAction] = []
    for up_idx, down_idx in pairs:
        up = net.gen.loc[up_idx]
        down = net.gen.loc[down_idx]
        up_headroom = _max_p(up) - float(up.p_mw)
        down_headroom = float(down.p_mw) - _min_p(down)
        feasible = min(up_headroom, down_headroom)
        if feasible <= 1e-6:
            continue
        for frac in config.action_space.redispatch_step_fracs:
            delta = min(feasible * frac, config.action_space.redispatch_max_mw_per_action)
            if delta <= 1e-6:
                continue
            target_bus = _nearest_affected_bus(int(up.bus), affected, graph)
            actions.append(
                CorrectiveAction(
                    action_type="gen_redispatch_pair",
                    target_table="gen",
                    target_index=(up_idx, down_idx),
                    params={"delta_mw": round(float(delta), 6)},
                    cost=abs(delta),
                    disruptive_rank=2,
                    reason=(
                        f"Redispatch +{delta:.2f} MW at generator {up_idx} and "
                        f"-{delta:.2f} MW at generator {down_idx} near bus {target_bus}"
                    ),
                )
            )
    return actions


def _gen_voltage_setpoint_actions(
    net: pp.pandapowerNet,
    affected: set[int],
    config: CorrectiveOptimizerConfig,
) -> list[CorrectiveAction]:
    if not len(net.gen) or "vm_pu" not in net.gen.columns:
        return []
    rows = _local_or_all(_in_service_rows(net.gen), affected)
    deltas = np.asarray(config.action_space.voltage_setpoint_deltas_pu, dtype=float)
    actions: list[CorrectiveAction] = []
    for row in rows.itertuples():
        idx = int(row.Index)
        current = getattr(row, "vm_pu", None)
        if not _is_finite_number(current):
            continue
        bus = int(row.bus)
        vmin, vmax = _bus_voltage_limits(net, bus)
        current_value = float(current)
        new_values = np.clip(current_value + deltas, vmin, vmax)
        valid_mask = ~np.isclose(new_values, current_value, atol=1e-5)
        for delta, new_vm in zip(deltas[valid_mask], new_values[valid_mask]):
            actions.append(
                CorrectiveAction(
                    "gen_voltage_setpoint", "gen", idx,
                    {"vm_pu": round(float(new_vm), 6), "delta_pu": float(delta)},
                    cost=abs(delta) * 100.0, disruptive_rank=1,
                    reason=f"Set generator {idx} voltage target to {new_vm:.3f} pu",
                )
            )
    return actions


def _ext_grid_voltage_setpoint_actions(
    net: pp.pandapowerNet,
    affected: set[int],
    config: CorrectiveOptimizerConfig,
) -> list[CorrectiveAction]:
    if not len(net.ext_grid) or "vm_pu" not in net.ext_grid.columns:
        return []
    rows = _local_or_all(_in_service_rows(net.ext_grid), affected)
    deltas = np.asarray(config.action_space.ext_grid_voltage_deltas_pu, dtype=float)
    actions: list[CorrectiveAction] = []
    for row in rows.itertuples():
        idx = int(row.Index)
        current = getattr(row, "vm_pu", None)
        if not _is_finite_number(current):
            continue
        bus = int(row.bus)
        vmin, vmax = _bus_voltage_limits(net, bus)
        current_value = float(current)
        new_values = np.clip(current_value + deltas, vmin, vmax)
        valid_mask = ~np.isclose(new_values, current_value, atol=1e-5)
        for delta, new_vm in zip(deltas[valid_mask], new_values[valid_mask]):
            actions.append(
                CorrectiveAction(
                    "ext_grid_voltage_setpoint", "ext_grid", idx,
                    {"vm_pu": round(float(new_vm), 6), "delta_pu": float(delta)},
                    cost=abs(delta) * 125.0, disruptive_rank=2,
                    reason=f"Set external grid {idx} voltage target to {new_vm:.3f} pu",
                )
            )
    return actions


def _tap_actions(
    net: pp.pandapowerNet,
    report: ViolationReport,
    affected: set[int],
    config: CorrectiveOptimizerConfig,
) -> list[CorrectiveAction]:
    if not len(net.trafo) or "tap_pos" not in net.trafo.columns:
        return []
    actions: list[CorrectiveAction] = []
    for idx, row in _in_service_rows(net.trafo).iterrows():
        if affected and int(row.hv_bus) not in affected and int(row.lv_bus) not in affected:
            continue
        tap_pos = row.get("tap_pos")
        tap_min = row.get("tap_min")
        tap_max = row.get("tap_max")
        if not _is_finite_number(tap_min) or not _is_finite_number(tap_max):
            continue
        if not _is_finite_number(tap_pos):
            if float(tap_min) <= 0.0 <= float(tap_max):
                tap_pos = 0
            else:
                continue
        for tap_delta in config.action_space.tap_step_deltas:
            new_tap = int(float(tap_pos) + tap_delta)
            if new_tap < int(float(tap_min)) or new_tap > int(float(tap_max)):
                continue
            actions.append(
                CorrectiveAction(
                    "trafo_tap_change", "trafo", int(idx),
                    {"tap_pos": new_tap, "tap_delta": tap_delta},
                    cost=abs(tap_delta) * 5.0, disruptive_rank=3,
                    reason=f"Move transformer {idx} tap to {new_tap}",
                )
            )
    return actions


def _line_switching_actions(
    net: pp.pandapowerNet,
    report: ViolationReport,
    graph: dict[int, set[int]],
    config: CorrectiveOptimizerConfig,
) -> list[CorrectiveAction]:
    if not len(net.line) or not config.action_space.line_switch_allowlist:
        return []
    actions: list[CorrectiveAction] = []
    outaged = set(int(idx) for idx in report.overloaded_lines.index)
    for idx in config.action_space.line_switch_allowlist:
        if idx not in net.line.index or idx in outaged:
            continue
        current = bool(net.line.at[idx, "in_service"]) if "in_service" in net.line.columns else True
        if current and not config.action_space.allow_islanding and not _line_removal_keeps_connected(net, int(idx)):
            continue
        actions.append(
            CorrectiveAction(
                "line_switch", "line", int(idx), {"in_service": not current},
                cost=250.0, disruptive_rank=8,
                reason=f"{'Close' if not current else 'Open'} allowlisted line {idx}",
            )
        )
    return actions


def _reactive_load_actions(
    net: pp.pandapowerNet,
    report: ViolationReport,
    affected: set[int],
    config: CorrectiveOptimizerConfig,
) -> list[CorrectiveAction]:
    if not len(net.load) or "q_mvar" not in net.load.columns:
        return []
    if report.low_voltage_buses.empty and report.high_voltage_buses.empty:
        return []
    rows = _local_or_all(_in_service_rows(net.load), affected)
    actions: list[CorrectiveAction] = []
    for idx, row in rows.iterrows():
        current_q = float(row.get("q_mvar", 0.0))
        base = abs(current_q)
        if base <= 1e-9:
            continue
        for frac in config.action_space.reactive_load_step_fracs:
            new_q = current_q + frac * base
            actions.append(
                CorrectiveAction(
                    "reactive_load_adjustment", "load", int(idx),
                    {"q_mvar": round(new_q, 6), "delta_q_mvar": round(new_q - current_q, 6)},
                    cost=abs(new_q - current_q) * 4.0, disruptive_rank=3,
                    reason=f"Adjust load {idx} reactive demand to {new_q:.2f} MVAr",
                )
            )
    return actions


def _load_curtailment_actions(
    net: pp.pandapowerNet,
    affected: set[int],
    config: CorrectiveOptimizerConfig,
) -> list[CorrectiveAction]:
    if not len(net.load):
        return []
    rows = _local_or_all(_in_service_rows(net.load), affected)
    total_load = float(rows["p_mw"].clip(lower=0.0).sum()) if "p_mw" in rows.columns else 0.0
    max_total = total_load * config.action_space.max_total_curtailment_frac
    fractions = np.asarray(config.action_space.load_curtailment_fracs, dtype=float)
    actions: list[CorrectiveAction] = []
    for row in rows.sort_values("p_mw", ascending=False).itertuples():
        idx = int(row.Index)
        p = max(0.0, float(getattr(row, "p_mw", 0.0)))
        q = float(getattr(row, "q_mvar", 0.0))
        if p <= 1e-9:
            continue
        curtailed = np.minimum(p * fractions, max_total)
        valid_mask = curtailed > 1e-9
        valid_curtail = curtailed[valid_mask]
        valid_fracs = fractions[valid_mask]
        if not len(valid_curtail):
            continue
        new_p_values = np.maximum(0.0, p - valid_curtail)
        scale_values = np.maximum(0.0, new_p_values / p)
        new_q_values = q * scale_values
        for frac, new_p, new_q in zip(valid_fracs, new_p_values, new_q_values):
            curtailed_p = p - float(new_p)
            actions.append(
                CorrectiveAction(
                    "load_curtailment", "load", idx,
                    {
                        "p_mw": round(float(new_p), 6),
                        "q_mvar": round(float(new_q), 6),
                        "curtail_p_mw": round(float(curtailed_p), 6),
                        "fraction": float(frac),
                    },
                    cost=curtailed_p * 1000.0, disruptive_rank=10,
                    reason=f"Curtail load {idx} by {curtailed_p:.2f} MW",
                )
            )
    return actions


def _choose_best_candidate(
    candidates: list[CandidateResult],
    start_score: ScoreBreakdown,
    seen_states: set[tuple] | None = None,
) -> CandidateResult | None:
    seen_states = seen_states or set()
    unseen = [candidate for candidate in candidates if _state_signature(candidate.net) not in seen_states]
    safe_candidates = [candidate for candidate in unseen if candidate.safe and candidate.converged]
    improving = [
        candidate for candidate in unseen
        if candidate.converged and candidate.score.score < start_score.score - 1e-6
    ]
    ranked_pool = safe_candidates or improving
    if not ranked_pool:
        return None
    return min(
        ranked_pool,
        key=lambda candidate: (
            candidate.score.remaining_violations,
            candidate.score.overload_amount,
            candidate.score.voltage_deviation,
            candidate.score.intervention_cost,
        ),
    )


def _deduplicate_sort_and_cap(
    actions: list[CorrectiveAction],
    config: CorrectiveOptimizerConfig,
    affected: set[int],
    graph: dict[int, set[int]],
) -> list[CorrectiveAction]:
    if config.action_space.candidate_dedup:
        deduped: dict[tuple, CorrectiveAction] = {}
        for action in actions:
            deduped.setdefault(action.signature(), action)
        actions = list(deduped.values())

    priority = {
        "gen_voltage_setpoint": 1,
        "ext_grid_voltage_setpoint": 2,
        "gen_redispatch_pair": 3,
        "trafo_tap_change": 4,
        "reactive_load_adjustment": 5,
        "line_switch": 8,
        "load_curtailment": 10,
    }
    actions.sort(key=lambda action: (priority.get(action.action_type, 99), action.disruptive_rank, action.cost, action.signature()))

    by_type: Counter[str] = Counter()
    capped: list[CorrectiveAction] = []
    for action in actions:
        type_limit = min(config.max_candidates_per_type, _candidate_limit_for_action_type(action.action_type, config))
        if by_type[action.action_type] >= type_limit:
            continue
        capped.append(action)
        by_type[action.action_type] += 1
        if len(capped) >= config.max_candidates_total:
            break
    return capped


def _candidate_limit_for_action_type(action_type: str, config: CorrectiveOptimizerConfig) -> int:
    limits = {
        "gen_redispatch_pair": config.max_gen_redispatch_candidates,
        "gen_voltage_setpoint": config.max_gen_voltage_setpoint_candidates,
        "trafo_tap_change": config.max_trafo_tap_candidates,
        "ext_grid_voltage_setpoint": config.max_ext_grid_voltage_candidates,
        "reactive_load_adjustment": config.max_reactive_load_candidates,
        "load_curtailment": config.max_load_curtailment_candidates,
        "line_switch": config.max_line_switch_candidates,
    }
    return max(0, limits.get(action_type, config.max_candidates_per_type))


def _build_bus_graph(net: pp.pandapowerNet) -> dict[int, set[int]]:
    graph = {int(idx): set() for idx in net.bus.index}
    for _, row in _in_service_rows(net.line).iterrows():
        a, b = int(row.from_bus), int(row.to_bus)
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    if len(net.trafo):
        for _, row in _in_service_rows(net.trafo).iterrows():
            a, b = int(row.hv_bus), int(row.lv_bus)
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
    return graph


def _affected_buses(
    net: pp.pandapowerNet,
    report: ViolationReport,
    graph: dict[int, set[int]],
    radius: int,
) -> set[int]:
    seeds = _stressed_buses(report)
    expanded: set[int] = set()
    for seed in seeds:
        expanded.update(_buses_within_radius(graph, seed, radius))
    return expanded


def _stressed_buses(report: ViolationReport) -> set[int]:
    buses: set[int] = set()
    if not report.overloaded_lines.empty:
        buses.update(report.overloaded_lines["from_bus"].astype(int).tolist())
        buses.update(report.overloaded_lines["to_bus"].astype(int).tolist())
    if not report.overloaded_trafos.empty:
        buses.update(report.overloaded_trafos["hv_bus"].astype(int).tolist())
        buses.update(report.overloaded_trafos["lv_bus"].astype(int).tolist())
    if not report.low_voltage_buses.empty:
        buses.update(int(idx) for idx in report.low_voltage_buses.index)
    if not report.high_voltage_buses.empty:
        buses.update(int(idx) for idx in report.high_voltage_buses.index)
    return buses


def _buses_within_radius(graph: dict[int, set[int]], start: int, radius: int) -> set[int]:
    seen = {int(start)}
    queue: deque[tuple[int, int]] = deque([(int(start), 0)])
    while queue:
        bus, dist = queue.popleft()
        if dist >= radius:
            continue
        for neighbor in graph.get(bus, set()):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, dist + 1))
    return seen


def _local_or_all(frame: pd.DataFrame, affected: set[int]) -> pd.DataFrame:
    if not affected or "bus" not in frame.columns:
        return frame
    local = frame[frame["bus"].astype(int).isin(affected)]
    return local if len(local) else frame


def _in_service_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "in_service" not in frame.columns:
        return frame
    return frame[frame["in_service"].fillna(True).astype(bool)]


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _min_p(row: pd.Series) -> float:
    value = row.get("min_p_mw", 0.0)
    return float(value) if _is_finite_number(value) else 0.0


def _max_p(row: pd.Series) -> float:
    value = row.get("max_p_mw", float(row.p_mw) * 1.4 + 1.0)
    return float(value) if _is_finite_number(value) else float(row.p_mw) * 1.4 + 1.0


def _bus_voltage_limits(net: pp.pandapowerNet, bus_idx: int) -> tuple[float, float]:
    vmin = net.bus.at[bus_idx, "min_vm_pu"] if "min_vm_pu" in net.bus.columns else 0.95
    vmax = net.bus.at[bus_idx, "max_vm_pu"] if "max_vm_pu" in net.bus.columns else 1.05
    vmin = float(vmin) if _is_finite_number(vmin) else 0.95
    vmax = float(vmax) if _is_finite_number(vmax) else 1.05
    return vmin, vmax


def _nearest_affected_bus(bus: int, affected: set[int], graph: dict[int, set[int]]) -> int | str:
    if not affected:
        return "any"
    distances = sorted((_graph_distance(graph, bus, target), target) for target in affected)
    return distances[0][1]


def _graph_distance(graph: dict[int, set[int]], start: int, target: int) -> int:
    if start == target:
        return 0
    queue: deque[tuple[int, int]] = deque([(start, 0)])
    seen = {start}
    while queue:
        node, dist = queue.popleft()
        for neighbor in graph.get(node, set()):
            if neighbor == target:
                return dist + 1
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, dist + 1))
    return 1_000_000


def _line_removal_keeps_connected(net: pp.pandapowerNet, line_idx: int) -> bool:
    trial = deepcopy(net)
    trial.line.at[line_idx, "in_service"] = False
    graph = _build_bus_graph(trial)
    active_buses = set(graph)
    if not active_buses:
        return False
    start = next(iter(active_buses))
    connected = _buses_within_radius(graph, start, len(active_buses) + 1)
    return active_buses.issubset(connected)


def _candidate_sort_key(choice: tuple[CorrectiveAction, pp.pandapowerNet, ViolationReport, float]) -> tuple:
    action, _, report, score = choice
    return (not report.is_safe, score, action.disruptive_rank, action.cost)


def _optimization_result(
    status: str,
    actions: list[CorrectiveAction],
    initial_score: float,
    final_score: float,
    initial_report: ViolationReport,
    final_report: ViolationReport,
    gridsfm_used: bool,
    gridsfm_skip_reason: str | None,
    step_logs: list[dict[str, Any]],
    net: pp.pandapowerNet,
) -> OptimizationResult:
    return OptimizationResult(
        status=status,
        actions=actions,
        initial_score=initial_score,
        final_score=final_score,
        initial_violations=initial_report,
        final_violations=final_report,
        gridsfm_used=gridsfm_used,
        gridsfm_skip_reason=gridsfm_skip_reason,
        step_logs=step_logs,
        net=deepcopy(net),
    )


def _state_signature(net: pp.pandapowerNet) -> tuple:
    def numeric_values(frame: pd.DataFrame, column: str) -> tuple[float, ...]:
        if not len(frame) or column not in frame.columns:
            return ()
        return tuple(round(float(value), 4) for value in frame[column].fillna(0.0).tolist())

    def service_values(frame: pd.DataFrame) -> tuple[bool, ...]:
        if not len(frame) or "in_service" not in frame.columns:
            return ()
        return tuple(bool(value) for value in frame["in_service"].tolist())

    return (
        numeric_values(net.gen, "p_mw"),
        numeric_values(net.gen, "vm_pu"),
        numeric_values(net.ext_grid, "vm_pu"),
        numeric_values(net.load, "p_mw"),
        numeric_values(net.load, "q_mvar"),
        numeric_values(net.trafo, "tap_pos"),
        service_values(net.line),
        service_values(net.trafo),
    )


def _hashable_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable_value(val)) for key, val in value.items()))
    if isinstance(value, list):
        return tuple(_hashable_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_hashable_value(item) for item in value)
    if isinstance(value, float):
        return round(value, 8)
    return value


def _observe(report: ViolationReport, power_flow_mode: PowerFlowMode) -> str:
    if not report.converged:
        return "Power flow did not converge after the contingency."
    if report.is_safe:
        if power_flow_mode == "ac":
            return "No thermal or voltage violations were detected by the AC power-flow screen."
        return "No thermal violations were detected by the DC power-flow screen."
    if power_flow_mode == "ac":
        return (
            f"Detected {len(report.overloaded_lines)} overloaded lines, "
            f"{len(report.overloaded_trafos)} overloaded transformers, "
            f"{len(report.low_voltage_buses)} low-voltage buses, and "
            f"{len(report.high_voltage_buses)} high-voltage buses."
        )
    return (
        f"Detected {len(report.overloaded_lines)} overloaded lines, "
        f"{len(report.overloaded_trafos)} overloaded transformers, and "
        "no voltage checks because DC power flow does not solve voltage magnitudes."
    )


def _think(report: ViolationReport, power_flow_mode: PowerFlowMode) -> str:
    if not report.converged:
        return "Try conservative interventions that improve power-flow solvability."
    if power_flow_mode == "ac" and (len(report.low_voltage_buses) or len(report.high_voltage_buses)):
        return "Primary issue includes voltage security, so test voltage setpoints, transformer taps, reactive demand changes, and load relief."
    if len(report.overloaded_lines) or len(report.overloaded_trafos):
        return "Primary issue is thermal loading, so test active-power redispatch and targeted load relief."
    if power_flow_mode == "ac":
        return "The contingency is AC-secure, so no corrective action is required."
    return "The contingency is DC-secure, so no corrective action is required."
