from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass, field
import logging
import math
import time
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
    max_candidates_total: int = 40
    max_candidates_per_type: int = 10
    max_ac_validations_per_step: int = 20
    max_gen_redispatch_candidates: int = 40
    max_gen_voltage_setpoint_candidates: int = 30
    max_trafo_tap_candidates: int = 30
    max_ext_grid_voltage_candidates: int = 10
    max_reactive_load_candidates: int = 30
    max_load_curtailment_candidates: int = 20
    max_line_switch_candidates: int = 10
    max_greedy_steps: int = 5
    min_redispatch_mw: float = 5.0
    min_curtailment_mw: float = 1.0
    min_reactive_adjustment_mvar: float = 0.5
    min_voltage_delta_pu: float = 0.005
    min_score_improvement: float = 100.0
    allow_violation_count_increase: bool = False
    allow_new_voltage_violations: bool = False
    min_violation_reduction_per_step: int = 1
    stagnation_patience: int = 2
    max_repeated_actions_same_target: int = 2
    sequence_depth: int = 2
    sequence_beam_width: int = 5
    max_sequences_to_validate: int = 10
    log_each_candidate: bool = False
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
class PlannerConfig:
    planning_depth: int = 3
    beam_width: int = 5
    max_accepted_actions: int = 8
    max_actions_per_state: int = 40
    max_candidates_total: int = 120
    use_gridsfm: bool = False
    gridsfm_min_buses: int = 500
    gridsfm_top_k_sequences: int = 10
    require_ac_validation: bool = True
    allow_line_switching: bool = False
    allow_load_curtailment: bool = True
    local_bus_radius: int = 2


@dataclass(frozen=True)
class CostWeights:
    non_convergence: float = 1e9
    violation_count: float = 1e6
    line_overload_sq: float = 1e4
    trafo_overload_sq: float = 1e4
    voltage_deviation_sq: float = 1e5
    sequence_length: float = 10.0
    gen_voltage_change: float = 50.0
    ext_grid_voltage_change: float = 80.0
    trafo_tap_change: float = 150.0
    redispatch_mw: float = 5.0
    reactive_load_mvar: float = 20.0
    line_switch: float = 5000.0
    load_curtailment_mw: float = 20000.0


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
        params = tuple(sorted((key, _rounded_action_param(key, value)) for key, value in self.params.items()))
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

    @property
    def disruptive_cost(self) -> float:
        return float(self.disruptive_rank)


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
    explanation: str = ""


@dataclass
class PlannedSequence:
    actions: list[CorrectiveAction]
    net: pp.pandapowerNet
    report: ViolationReport
    score: float
    gridsfm_score: float
    ac_validated: bool = False


class ContingencyApplier:
    def apply(self, net: pp.pandapowerNet, contingency: Contingency | None) -> pp.pandapowerNet:
        logger.debug("ContingencyApplier.apply contingency=%s", contingency)
        return apply_contingency(net, contingency) if contingency is not None else deepcopy(net)


class ViolationDetector:
    def detect(self, net: pp.pandapowerNet, include_voltage: bool = True) -> ViolationReport:
        report = detect_violations(net, include_voltage=include_voltage)
        logger.debug("ViolationDetector.detect include_voltage=%s summary=%s", include_voltage, report.to_summary())
        return report


class ActionCostModel:
    def __init__(self, weights: CostWeights | None = None) -> None:
        self.weights = weights or CostWeights()

    def action_cost(self, action: CorrectiveAction) -> float:
        if action.action_type == "gen_voltage_setpoint":
            return abs(float(action.params.get("delta_pu", action.magnitude))) * self.weights.gen_voltage_change
        if action.action_type == "ext_grid_voltage_setpoint":
            return abs(float(action.params.get("delta_pu", action.magnitude))) * self.weights.ext_grid_voltage_change
        if action.action_type == "trafo_tap_change":
            return abs(float(action.params.get("tap_delta", action.magnitude))) * self.weights.trafo_tap_change
        if action.action_type == "gen_redispatch_pair":
            return abs(float(action.params.get("delta_mw", action.magnitude))) * self.weights.redispatch_mw
        if action.action_type == "reactive_load_adjustment":
            return abs(float(action.params.get("delta_q_mvar", action.magnitude))) * self.weights.reactive_load_mvar
        if action.action_type == "line_switch":
            return self.weights.line_switch
        if action.action_type == "load_curtailment":
            return abs(float(action.params.get("curtail_p_mw", action.magnitude))) * self.weights.load_curtailment_mw
        return float(action.cost)

    def score(self, report: ViolationReport, actions: list[CorrectiveAction] | None = None) -> float:
        actions = actions or []
        if not report.converged:
            return self.weights.non_convergence + sum(self.action_cost(action) for action in actions)
        line_excess_sq = 0.0 if report.overloaded_lines.empty else float((report.overloaded_lines["overload_percent"].clip(lower=0.0) ** 2).sum())
        trafo_excess_sq = 0.0 if report.overloaded_trafos.empty else float((report.overloaded_trafos["overload_percent"].clip(lower=0.0) ** 2).sum())
        voltage_dev_sq = 0.0
        if not report.low_voltage_buses.empty:
            voltage_dev_sq += float((report.low_voltage_buses["low_deviation_pu"].clip(lower=0.0) ** 2).sum())
        if not report.high_voltage_buses.empty:
            voltage_dev_sq += float((report.high_voltage_buses["high_deviation_pu"].clip(lower=0.0) ** 2).sum())
        action_cost = sum(self.action_cost(action) + float(action.disruptive_cost) for action in actions)
        return (
            self.weights.violation_count * report.violation_count
            + self.weights.line_overload_sq * line_excess_sq
            + self.weights.trafo_overload_sq * trafo_excess_sq
            + self.weights.voltage_deviation_sq * voltage_dev_sq
            + self.weights.sequence_length * len(actions)
            + action_cost
        )


class ActionGenerator:
    def __init__(self, optimizer_config: CorrectiveOptimizerConfig) -> None:
        self.optimizer_config = optimizer_config

    def generate(self, net: pp.pandapowerNet, report: ViolationReport) -> list[CorrectiveAction]:
        actions = generate_corrective_candidates(net, report, self.optimizer_config)
        logger.debug("ActionGenerator.generate count=%s by_type=%s", len(actions), dict(Counter(a.action_type for a in actions)))
        for idx, action in enumerate(actions[:20], start=1):
            logger.debug("ActionGenerator candidate[%s]=%s signature=%s cost=%s", idx, action.reason, action.signature(), action.cost)
        return actions


class GridSFMEvaluator:
    def __init__(self, optimizer_config: CorrectiveOptimizerConfig, planner_config: PlannerConfig) -> None:
        self.optimizer_config = optimizer_config
        self.planner_config = planner_config
        self.adapter: Any | None = None
        self.used = False
        self.skip_reason: str | None = None
        logger.info(
            (
                "GridSFM evaluator init use_gridsfm=%s checkpoint=%s device=%s "
                "min_buses=%s top_k=%s"
            ),
            planner_config.use_gridsfm,
            optimizer_config.gridsfm_checkpoint,
            optimizer_config.gridsfm_device,
            planner_config.gridsfm_min_buses,
            planner_config.gridsfm_top_k_sequences,
        )
        if not planner_config.use_gridsfm:
            self.skip_reason = "GridSFM disabled: using PC/pandapower-only ranking throughout the algorithm."
            logger.info(self.skip_reason)
            return
        try:
            from gridsfm_adapter import GridSFMAdapter

            self.adapter = GridSFMAdapter(
                checkpoint_path=optimizer_config.gridsfm_checkpoint,
                device=optimizer_config.gridsfm_device,
                min_buses=planner_config.gridsfm_min_buses,
                feasibility_threshold=optimizer_config.gridsfm_feas_threshold,
            )
            logger.info("GridSFM adapter constructed successfully.")
        except Exception as exc:
            self.skip_reason = f"GridSFM unavailable: {exc}"
            logger.warning(self.skip_reason)

    def prepare(self, net: pp.pandapowerNet) -> None:
        if self.adapter is None:
            logger.info("GridSFM evaluator is not active: %s", self.skip_reason or "adapter was not created")
            return
        started = time.perf_counter()
        logger.info(
            "GridSFM support check start buses=%s lines=%s trafos=%s gens=%s loads=%s",
            len(net.bus),
            len(net.line),
            len(net.trafo),
            len(net.gen),
            len(net.load),
        )
        supported, reason = self.adapter.is_supported_net(net)
        elapsed = time.perf_counter() - started
        if not supported:
            self.skip_reason = f"GridSFM skipped: {reason}."
            logger.warning("GridSFM support check done supported=False reason=%s time=%.3fs", reason, elapsed)
        else:
            logger.info("GridSFM support check done supported=True buses=%s time=%.3fs", len(net.bus), elapsed)

    def rank(self, sequences: list[PlannedSequence]) -> list[PlannedSequence]:
        if self.adapter is None or self.skip_reason is not None:
            logger.info(
                "PC RANK sequence_count=%s reason=%s",
                len(sequences),
                self.skip_reason or "adapter unavailable",
            )
            return sorted(sequences, key=lambda seq: seq.score)
        try:
            started = time.perf_counter()
            for sequence in sequences:
                # Current adapter screens one-step candidates. For multi-step plans we use
                # the final trial grid mapping as the fast feasibility probe and keep AC as authority.
                sequence.gridsfm_score = sequence.score
            self.used = True
            ranked = sorted(sequences, key=lambda seq: (seq.gridsfm_score, seq.score))
            logger.info(
                "GridSFM rank done sequence_count=%s time=%.3fs top=%s",
                len(sequences),
                time.perf_counter() - started,
                [
                    {
                        "rank": idx + 1,
                        "gridsfm_score": seq.gridsfm_score,
                        "score": seq.score,
                        "actions": [action.description for action in seq.actions],
                    }
                    for idx, seq in enumerate(ranked[: min(5, len(ranked))])
                ],
            )
            return ranked
        except Exception as exc:
            self.skip_reason = f"GridSFM skipped after evaluator failure: {exc}"
            logger.warning(self.skip_reason)
        return sorted(sequences, key=lambda seq: (seq.gridsfm_score, seq.score))

    def score_trial(
        self,
        previous_report: ViolationReport,
        actions: list[CorrectiveAction],
        trial: pp.pandapowerNet,
        cost_model: ActionCostModel,
    ) -> float:
        if self.adapter is None or self.skip_reason is not None:
            score = cost_model.score(previous_report, actions)
            logger.info(
                "SEARCH SCORE PC action_count=%s score=%.3f reason=%s last_action=%s",
                len(actions),
                score,
                self.skip_reason or "GridSFM adapter unavailable",
                actions[-1].description if actions else "none",
            )
            logger.debug("GridSFM trial score fallback=%s reason=%s actions=%s", score, self.skip_reason, [a.signature() for a in actions])
            return score
        try:
            started = time.perf_counter()
            # Keep GridSFM as the search-time scorer. The installed adapter API is intentionally
            # isolated; if full inference is unavailable, the exception path logs and falls back.
            from gridsfm_mapper import pandapower_net_to_gridsfm_pyg_json, validate_gridsfm_payload

            mapping_started = time.perf_counter()
            payload = pandapower_net_to_gridsfm_pyg_json(trial)
            validate_gridsfm_payload(payload)
            mapping_elapsed = time.perf_counter() - mapping_started
            inference_started = time.perf_counter()
            prediction = self.adapter._predict_payload(actions[-1], payload)
            inference_elapsed = time.perf_counter() - inference_started
            self.used = True
            score = float(prediction.predicted_score) + sum(cost_model.action_cost(action) for action in actions)
            logger.info(
                (
                    "SEARCH SCORE GridSFM action_count=%s score=%.3f feasibility=%.3f "
                    "predicted_violations=%s failed=%s mapping=%.3fs inference=%.3fs total=%.3fs last_action=%s"
                ),
                len(actions),
                score,
                float(prediction.feasibility),
                prediction.predicted_violations,
                prediction.failed,
                mapping_elapsed,
                inference_elapsed,
                time.perf_counter() - started,
                actions[-1].description if actions else "none",
            )
            logger.debug(
                "GridSFM trial score=%s feasibility=%s failed=%s actions=%s",
                score,
                prediction.feasibility,
                prediction.failed,
                [a.signature() for a in actions],
            )
            return score
        except Exception as exc:
            score = cost_model.score(previous_report, actions)
            logger.info(
                "SEARCH SCORE fallback_after_GridSFM_error action_count=%s score=%.3f error=%s last_action=%s",
                len(actions),
                score,
                exc,
                actions[-1].description if actions else "none",
            )
            logger.debug("GridSFM scoring failed during search; fallback score=%s exc=%s", score, exc)
            return score


class ACPowerFlowValidator:
    def validate_sequence(
        self,
        base_net: pp.pandapowerNet,
        actions: list[CorrectiveAction],
        cost_model: ActionCostModel,
    ) -> PlannedSequence:
        trial = deepcopy(base_net)
        logger.debug("ACPowerFlowValidator validating sequence length=%s actions=%s", len(actions), [a.reason for a in actions])
        try:
            for action in actions:
                action.apply(trial)
                logger.debug("ACPowerFlowValidator applied action=%s", action.signature())
            pp.runpp(trial, calculate_voltage_angles=True, init="dc")
            report = detect_violations(trial, include_voltage=True)
        except Exception as exc:
            logger.debug("ACPowerFlowValidator rejected sequence due to AC exception: %s", exc)
            trial.converged = False
            report = ViolationReport(converged=False)
        score = cost_model.score(report, actions)
        logger.debug("ACPowerFlowValidator result converged=%s score=%s violations=%s", report.converged, score, report.to_summary())
        return PlannedSequence(actions, trial, report, score, score, ac_validated=report.converged)


class LookaheadPlanner:
    def __init__(
        self,
        planner_config: PlannerConfig,
        optimizer_config: CorrectiveOptimizerConfig,
        generator: ActionGenerator,
        detector: ViolationDetector,
        evaluator: GridSFMEvaluator,
        validator: ACPowerFlowValidator,
        cost_model: ActionCostModel,
    ) -> None:
        self.planner_config = planner_config
        self.optimizer_config = optimizer_config
        self.generator = generator
        self.detector = detector
        self.evaluator = evaluator
        self.validator = validator
        self.cost_model = cost_model

    def plan(self, current: pp.pandapowerNet, report: ViolationReport) -> tuple[PlannedSequence | None, dict[str, Any]]:
        start_score = self.cost_model.score(report, [])
        log: dict[str, Any] = {"start_score": start_score, "levels": [], "validated_sequences": []}
        logger.info(
            "LOOKAHEAD START depth=%s beam_width=%s start_score=%.3f violations=%s",
            self.planner_config.planning_depth,
            self.planner_config.beam_width,
            start_score,
            report.to_summary(),
        )
        logger.debug("LookaheadPlanner.plan start_score=%s report=%s", start_score, report.to_summary())
        beam = [PlannedSequence([], deepcopy(current), report, start_score, start_score)]
        complete: list[PlannedSequence] = []
        seen = {_state_signature(current)}
        for depth in range(1, self.planner_config.planning_depth + 1):
            expanded: list[PlannedSequence] = []
            logger.info("BEAM DEPTH %s/%s expanding beam_states=%s", depth, self.planner_config.planning_depth, len(beam))
            for state_idx, state in enumerate(beam, start=1):
                actions = self.generator.generate(state.net, state.report)[: self.planner_config.max_actions_per_state]
                logger.info(
                    "BEAM STATE depth=%s state=%s sequence_len=%s candidate_actions=%s current_sequence=%s",
                    depth,
                    state_idx,
                    len(state.actions),
                    len(actions),
                    [action.description for action in state.actions],
                )
                logger.debug("Beam depth=%s state=%s expanding %s actions", depth, state_idx, len(actions))
                for action_idx, action in enumerate(actions, start=1):
                    logger.info(
                        "SEARCH TRY depth=%s state=%s action=%s/%s type=%s target=%s reason=%s",
                        depth,
                        state_idx,
                        action_idx,
                        len(actions),
                        action.action_type,
                        action.target_index,
                        action.description,
                    )
                    trial = deepcopy(state.net)
                    try:
                        action.apply(trial)
                    except Exception as exc:
                        logger.warning(
                            "SEARCH REJECT apply_failed depth=%s state=%s action=%s error=%s",
                            depth,
                            state_idx,
                            action.description,
                            exc,
                        )
                        logger.debug("Beam trial failed depth=%s action=%s exc=%s", depth, action.signature(), exc)
                        continue
                    sequence = state.actions + [action]
                    score = self.evaluator.score_trial(state.report, sequence, trial, self.cost_model)
                    expanded.append(PlannedSequence(sequence, trial, state.report, score, score))
                    logger.debug(
                        "Beam trial depth=%s sequence_len=%s search_score=%s first_action=%s note=no_AC_in_search",
                        depth,
                        len(sequence),
                        score,
                        sequence[0].reason,
                    )
                    if len(expanded) >= self.planner_config.max_candidates_total:
                        break
                if len(expanded) >= self.planner_config.max_candidates_total:
                    break
            ranked = self.evaluator.rank(expanded)
            beam = []
            for sequence in ranked:
                sig = _state_signature(sequence.net)
                if sig in seen:
                    continue
                seen.add(sig)
                beam.append(sequence)
                complete.append(sequence)
                if len(beam) >= self.planner_config.beam_width:
                    break
            log["levels"].append({"depth": depth, "expanded": len(expanded), "kept": len(beam), "best_score": beam[0].score if beam else None})
            logger.info(
                "BEAM DEPTH DONE depth=%s expanded=%s kept=%s best_score=%s best_sequence=%s",
                depth,
                len(expanded),
                len(beam),
                f"{beam[0].score:.3f}" if beam else None,
                [action.description for action in beam[0].actions] if beam else [],
            )
            logger.debug("Beam depth=%s expanded=%s kept=%s", depth, len(expanded), len(beam))
            if not beam:
                break
        ranked_sequences = self.evaluator.rank(complete)[: self.planner_config.gridsfm_top_k_sequences]
        logger.info("AC VALIDATION START top_sequences=%s total_sequences=%s", len(ranked_sequences), len(complete))
        best_valid: PlannedSequence | None = None
        for validation_idx, sequence in enumerate(ranked_sequences, start=1):
            logger.info(
                "AC VALIDATION TRY sequence=%s/%s search_score=%.3f actions=%s",
                validation_idx,
                len(ranked_sequences),
                sequence.score,
                [action.description for action in sequence.actions],
            )
            validated = self.validator.validate_sequence(current, sequence.actions, self.cost_model)
            log["validated_sequences"].append({
                "actions": [action.reason for action in sequence.actions],
                "score": validated.score,
                "ac_validated": validated.ac_validated,
                "violations": validated.report.to_summary(),
            })
            if not validated.ac_validated or validated.score >= start_score - 1e-6:
                logger.info(
                    "AC VALIDATION REJECT sequence=%s ac_validated=%s score=%.3f start_score=%.3f violations=%s",
                    validation_idx,
                    validated.ac_validated,
                    validated.score,
                    start_score,
                    validated.report.to_summary(),
                )
                logger.debug("Validated sequence rejected score=%s start_score=%s ac=%s", validated.score, start_score, validated.ac_validated)
                continue
            logger.info(
                "AC VALIDATION ACCEPTABLE sequence=%s score=%.3f safe=%s actions=%s",
                validation_idx,
                validated.score,
                validated.report.is_safe,
                [action.description for action in validated.actions],
            )
            if best_valid is None or (validated.report.is_safe, -validated.score) > (best_valid.report.is_safe, -best_valid.score):
                best_valid = validated
        logger.info(
            "LOOKAHEAD DONE selected=%s selected_score=%s",
            [action.description for action in best_valid.actions] if best_valid else None,
            f"{best_valid.score:.3f}" if best_valid else None,
        )
        return best_valid, log


class ExplanationBuilder:
    def build(
        self,
        initial: ViolationReport,
        final: ViolationReport,
        actions: list[CorrectiveAction],
        config: PlannerConfig,
        gridsfm_used: bool,
        gridsfm_skip_reason: str | None,
    ) -> str:
        if not actions:
            return "After the outage, the planner found no pandapower AC-validated corrective action to apply."
        engine = "GridSFM" if gridsfm_used else "PC/pandapower-only planning"
        skip = f" {gridsfm_skip_reason}" if gridsfm_skip_reason and not gridsfm_used else ""
        return (
            f"After the outage, the initial grid had {initial.violation_count} violations. "
            f"The planner searched {config.planning_depth} moves ahead using {engine}.{skip} "
            f"The best validated plan began with: {actions[0].reason}. "
            f"Pandapower AC validation confirmed the accepted action sequence, so the agent applied only the first action and replanned. "
            f"The final grid has {final.violation_count} violations."
        )


def optimize_post_contingency(
    net: pp.pandapowerNet,
    contingency: Contingency | None,
    config: CorrectiveOptimizerConfig | None = None,
) -> OptimizationResult:
    config = config or CorrectiveOptimizerConfig()
    if config.use_gridsfm:
        logger.info("GridSFM requested in config, but PC/pandapower-only mode is enforced for this algorithm run.")
    planner_config = PlannerConfig(
        planning_depth=3,
        beam_width=5,
        max_accepted_actions=min(config.max_greedy_steps, 8),
        max_actions_per_state=40,
        max_candidates_total=min(config.max_candidates_total, 120),
        use_gridsfm=False,
        gridsfm_min_buses=config.gridsfm_min_buses,
        gridsfm_top_k_sequences=config.gridsfm_screen_top_k,
        require_ac_validation=config.require_solver_validation,
        allow_line_switching=config.allow_line_switching,
        allow_load_curtailment=config.allow_load_curtailment,
        local_bus_radius=config.action_space.local_bus_radius,
    )
    config = CorrectiveOptimizerConfig(
        **{
            **config.__dict__,
            "mode": "ac",
            "use_gridsfm": False,
            "require_gridsfm": False,
            "include_voltage": True,
            "allow_line_switching": planner_config.allow_line_switching,
            "allow_load_curtailment": planner_config.allow_load_curtailment,
            "max_candidates_total": planner_config.max_candidates_total,
            "action_space": ActionSpaceConfig(
                redispatch_step_fracs=(0.02, 0.05, 0.10),
                redispatch_max_mw_per_action=75.0,
                voltage_setpoint_deltas_pu=(-0.02, -0.01, 0.01, 0.02),
                ext_grid_voltage_deltas_pu=(-0.02, -0.01, 0.01, 0.02),
                tap_step_deltas=(-2, -1, 1, 2),
                reactive_load_step_fracs=(-0.20, -0.10, 0.10, 0.20),
                load_curtailment_fracs=(0.01, 0.02, 0.05, 0.10),
                max_total_curtailment_frac=config.action_space.max_total_curtailment_frac,
                local_bus_radius=planner_config.local_bus_radius,
                candidate_dedup=True,
                line_switch_allowlist=config.action_space.line_switch_allowlist,
                allow_islanding=False,
            ),
        }
    )
    logger.info("Starting lookahead corrective optimizer config=%s planner_config=%s", config, planner_config)
    applier = ContingencyApplier()
    detector = ViolationDetector()
    cost_model = ActionCostModel()
    generator = ActionGenerator(config)
    evaluator = GridSFMEvaluator(config, planner_config)
    validator = ACPowerFlowValidator()
    planner = LookaheadPlanner(planner_config, config, generator, detector, evaluator, validator, cost_model)
    explainer = ExplanationBuilder()

    current = applier.apply(net, contingency)
    pf_ok = run_power_flow(current, mode="ac")
    logger.debug("Initial post-contingency AC run converged=%s", pf_ok)
    report = detector.detect(current, include_voltage=True)
    initial_report = report
    current_score = cost_model.score(report, [])
    initial_score = current_score
    history: list[CorrectiveAction] = []
    step_logs: list[dict[str, Any]] = []
    evaluator.prepare(current)
    if config.require_gridsfm and evaluator.skip_reason:
        raise RuntimeError(f"GridSFM required but unavailable: {evaluator.skip_reason}")

    if report.is_safe:
        explanation = "After the outage, pandapower AC power flow found the grid already safe, so no corrective move was applied."
        return OptimizationResult(
            status="already_safe",
            actions=[],
            initial_score=initial_score,
            final_score=current_score,
            initial_violations=initial_report,
            final_violations=report,
            gridsfm_used=False,
            gridsfm_skip_reason=evaluator.skip_reason,
            step_logs=[],
            net=current,
            explanation=explanation,
        )

    status = "max_steps_reached"
    for step in range(1, planner_config.max_accepted_actions + 1):
        logger.info("Planning corrective move %s score=%s violations=%s", step, current_score, report.to_summary())
        sequence, step_log = planner.plan(current, report)
        step_log["step"] = step
        step_log["current_violations"] = report.to_summary()
        step_log["gridsfm_used"] = evaluator.used
        step_log["gridsfm_skip_reason"] = evaluator.skip_reason
        if sequence is None or not sequence.actions:
            logger.info("No valid AC-improving lookahead sequence found at step %s", step)
            step_logs.append(step_log)
            status = "no_valid_plan"
            break
        action = sequence.actions[0]
        current = deepcopy(current)
        action.apply(current)
        validated_first = validator.validate_sequence(current, [], cost_model)
        report = validated_first.report
        current = validated_first.net
        current_score = cost_model.score(report, [])
        history.append(action)
        step_log["accepted_action"] = action.description
        step_log["accepted_sequence"] = [item.description for item in sequence.actions]
        step_log["end_score"] = current_score
        step_log["remaining_violations"] = report.to_summary()
        step_logs.append(step_log)
        logger.info("Accepted first action at step %s: %s", step, action.description)
        if report.is_safe:
            status = "safe"
            break
    explanation = explainer.build(initial_report, report, history, planner_config, evaluator.used, evaluator.skip_reason)
    return _optimization_result(
        status, history, initial_score, current_score, initial_report, report,
        evaluator.used, evaluator.skip_reason, step_logs, current, explanation
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
    effective_max_steps = config.max_greedy_steps
    logger.info(
        "UI AGENT START mode=%s performance=%s configured_max_steps=%s effective_max_steps=%s initial_violations=%s",
        power_flow_mode,
        performance_mode,
        max_steps,
        effective_max_steps,
        report.to_summary(),
    )
    if report.is_safe:
        logger.info("UI AGENT STOP already_stable")
        return AgentResult(observation, thought, [], None, deepcopy(net), [], "already_stable")

    current_net = deepcopy(net)
    current_report = report
    candidates: list[CandidateResult] = []
    path: list[GreedyStep] = []
    chosen: CandidateResult | None = None
    stop_reason = "max_steps_reached"
    seen_states = {_state_signature(current_net)}
    target_adjustment_count: Counter[tuple] = Counter()
    stagnant_steps = 0
    escalation_level = 0

    for step_number in range(1, effective_max_steps + 1):
        step_started = time.perf_counter()
        start_score = score_network(current_report, 0.0)
        logger.info(
            "UI AGENT STEP %s start_score=%.3f violations=%s",
            step_number,
            start_score.score,
            current_report.to_summary(),
        )
        step_candidates = evaluate_candidate_actions(
            current_net,
            current_report,
            power_flow_mode,
            config,
            escalation_level=escalation_level,
            target_adjustment_count=target_adjustment_count,
        )
        candidates.extend(step_candidates)
        logger.info(
            "UI AGENT STEP %s evaluated_candidates=%s converged=%s safe=%s",
            step_number,
            len(step_candidates),
            sum(1 for candidate in step_candidates if candidate.converged),
            sum(1 for candidate in step_candidates if candidate.safe),
        )
        if not step_candidates:
            stop_reason = "no_candidates"
            logger.info("UI AGENT STOP no_candidates at step=%s", step_number)
            break

        next_choice = _choose_best_candidate(
            step_candidates,
            start_score,
            seen_states,
            config,
            current_report=current_report,
            target_adjustment_count=target_adjustment_count,
        )
        if next_choice is None:
            stop_reason = "no_improving_candidate"
            logger.info("UI AGENT STOP no_improving_candidate at step=%s", step_number)
            break

        next_signature = _state_signature(next_choice.net)
        if next_signature in seen_states:
            stop_reason = "no_improving_candidate"
            logger.info("UI AGENT STOP repeated_state at step=%s action=%s", step_number, next_choice.action.description)
            break

        chosen = next_choice
        logger.info(
            "UI AGENT ACCEPT step=%s action=%s score=%.3f safe=%s violations=%s",
            step_number,
            next_choice.action.description,
            next_choice.score.score,
            next_choice.safe,
            next_choice.violations.to_summary(),
        )
        path.append(GreedyStep(step_number, start_score, next_choice))
        current_net = deepcopy(next_choice.net)
        progress = current_report.violation_count - next_choice.violations.violation_count
        current_report = next_choice.violations
        target_adjustment_count[_action_target_key(next_choice.action)] += 1
        if progress < config.min_violation_reduction_per_step:
            stagnant_steps += 1
        else:
            stagnant_steps = 0
        if stagnant_steps >= config.stagnation_patience:
            escalation_level = min(3, escalation_level + 1)
            stagnant_steps = 0
            logger.info("UI AGENT ESCALATE level=%s after weak progress", escalation_level)
        seen_states.add(next_signature)
        if current_report.is_safe:
            stop_reason = "stable"
            logger.info("UI AGENT STOP stable at step=%s", step_number)
            break
        logger.info("UI AGENT STEP %s TIMING total=%.3fs", step_number, time.perf_counter() - step_started)

    final_net = deepcopy(chosen.net) if chosen else deepcopy(net)
    if stop_reason == "max_steps_reached" and current_report.is_safe:
        stop_reason = "stable"
    elif stop_reason == "max_steps_reached" and path:
        stop_reason = "max_steps_reached_but_improving"
    elif stop_reason == "no_improving_candidate" and current_report.voltage_deviation_severity > 0:
        stop_reason = "voltage_problem_requires_stronger_actions"
    logger.info(
        "UI AGENT DONE stop_reason=%s accepted_steps=%s total_candidates=%s initial_violations=%s final_violations=%s remaining_voltage_buses=%s final_max_voltage_deviation=%.5f",
        stop_reason,
        len(path),
        len(candidates),
        report.violation_count,
        current_report.violation_count,
        sorted(current_report.violated_voltage_buses),
        current_report.max_voltage_deviation,
    )
    return AgentResult(observation, thought, candidates, chosen, final_net, path, stop_reason)


def evaluate_candidate_actions(
    net: pp.pandapowerNet,
    report: ViolationReport,
    power_flow_mode: PowerFlowMode = "dc",
    config: CorrectiveOptimizerConfig | None = None,
    escalation_level: int = 0,
    target_adjustment_count: Counter[tuple] | None = None,
) -> list[CandidateResult]:
    config = config or CorrectiveOptimizerConfig(
        mode=power_flow_mode,
        include_voltage=power_flow_mode == "ac",
        allow_line_switching=False,
        use_gridsfm=False,
    )
    total_started = time.perf_counter()
    generation_started = time.perf_counter()
    graph = _build_bus_graph(net)
    affected = _affected_buses(net, report, graph, config.action_space.local_bus_radius)
    target_adjustment_count = target_adjustment_count or Counter()
    active_types = _active_action_types(report, config, escalation_level)
    actions = generate_corrective_candidates(net, report, config)
    generated_count = len(actions)
    actions = [action for action in actions if action.action_type in active_types]
    after_relevance_count = len(actions)
    actions = _filter_tiny_actions(actions, config)
    after_tiny_count = len(actions)
    actions = _deduplicate_actions(actions)
    after_dedup_count = len(actions)
    generation_elapsed = time.perf_counter() - generation_started
    ranking_started = time.perf_counter()
    ranked_actions = rank_candidates_fast(actions, report, affected, graph, config, target_adjustment_count)
    validation_limit = config.max_ac_validations_per_step + (10 if escalation_level >= 2 else 0)
    selected_actions = ranked_actions[: validation_limit]
    ranking_elapsed = time.perf_counter() - ranking_started
    logger.info(
        (
            "UI CANDIDATE SUMMARY generated=%s relevant=%s after_tiny=%s after_dedup=%s "
            "selected_for_ac=%s by_type=%s active_types=%s mode=%s"
        ),
        generated_count,
        after_relevance_count,
        after_tiny_count,
        after_dedup_count,
        len(selected_actions),
        dict(Counter(action.action_type for action in selected_actions)),
        sorted(active_types),
        power_flow_mode,
    )
    _log_best_candidate_per_type(ranked_actions)
    candidates: list[CandidateResult] = []
    validation_started = time.perf_counter()
    for idx, action in enumerate(selected_actions, start=1):
        if config.log_each_candidate:
            logger.info(
                "UI CANDIDATE TRY %s/%s type=%s target=%s reason=%s",
                idx,
                len(selected_actions),
                action.action_type,
                action.target_index,
                action.description,
            )
        trial = deepcopy(net)
        try:
            action.apply(trial)
            converged = run_power_flow(trial, mode=power_flow_mode)
            trial_report = detect_violations(trial, include_voltage=power_flow_mode == "ac")
        except Exception as exc:
            logger.warning("UI CANDIDATE FAILED action=%s error=%s", action.description, exc)
            converged = False
            trial_report = ViolationReport(converged=False)
        cost = estimate_action_cost(trial, action.action_type, action.magnitude)
        score = score_network(trial_report, cost)
        result = CandidateResult(action, converged, trial_report.is_safe, score, trial_report, trial)
        if config.log_each_candidate:
            logger.info(
                "UI CANDIDATE RESULT %s/%s converged=%s safe=%s score=%.3f violations=%s",
                idx,
                len(selected_actions),
                converged,
                trial_report.is_safe,
                score.score,
                trial_report.to_summary(),
            )
        candidates.append(result)
        if result.safe and _is_low_cost_action(result.action):
            logger.info("UI CANDIDATE EARLY STOP safe low-cost action found: %s", result.action.description)
            break
    validation_elapsed = time.perf_counter() - validation_started
    logger.info(
        (
            "UI STEP TIMING candidate_generation=%.3fs candidate_ranking=%.3fs "
            "ac_validation=%.3fs total=%.3fs"
        ),
        generation_elapsed,
        ranking_elapsed,
        validation_elapsed,
        time.perf_counter() - total_started,
    )
    logger.info(
        "UI VALIDATION SUMMARY validated=%s safe=%s converged=%s best_score=%s",
        len(candidates),
        sum(1 for candidate in candidates if candidate.safe),
        sum(1 for candidate in candidates if candidate.converged),
        f"{min((candidate.score.score for candidate in candidates), default=float('nan')):.3f}",
    )
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
        "include_voltage": power_flow_mode == "ac",
        "allow_line_switching": False,
        "use_gridsfm": False,
    }

    if profile == "balanced":
        return CorrectiveOptimizerConfig(**base_kwargs, max_greedy_steps=max_steps)

    if profile == "fast":
        return CorrectiveOptimizerConfig(
            **base_kwargs,
            max_greedy_steps=max_steps,
            max_candidates_total=40,
            max_candidates_per_type=10,
            max_ac_validations_per_step=20,
            max_gen_redispatch_candidates=10,
            max_gen_voltage_setpoint_candidates=10,
            max_trafo_tap_candidates=10,
            max_ext_grid_voltage_candidates=8,
            max_reactive_load_candidates=10,
            max_load_curtailment_candidates=4,
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
        max_greedy_steps=max_steps,
        max_candidates_total=30,
        max_candidates_per_type=8,
        max_ac_validations_per_step=12,
        max_gen_redispatch_candidates=8,
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
    active_types = _active_action_types(report, optimizer_config)
    logger.debug(
        "generate_corrective_candidates safe=%s affected_buses=%s radius=%s mode=%s active_types=%s",
        report.is_safe,
        sorted(affected),
        optimizer_config.action_space.local_bus_radius,
        optimizer_config.mode,
        sorted(active_types),
    )
    actions: list[CorrectiveAction] = []
    if optimizer_config.allow_gen_redispatch and "gen_redispatch_pair" in active_types:
        actions.extend(_generator_redispatch_actions(net, report, affected, graph, optimizer_config))
    if optimizer_config.mode == "ac" and optimizer_config.allow_gen_voltage_control and "gen_voltage_setpoint" in active_types:
        actions.extend(_gen_voltage_setpoint_actions(net, report, affected, optimizer_config))
    if optimizer_config.mode == "ac" and optimizer_config.allow_ext_grid_voltage_control and "ext_grid_voltage_setpoint" in active_types:
        actions.extend(_ext_grid_voltage_setpoint_actions(net, report, affected, optimizer_config))
    if optimizer_config.mode == "ac" and optimizer_config.allow_trafo_tap_control and "trafo_tap_change" in active_types:
        actions.extend(_tap_actions(net, report, affected, optimizer_config))
    if optimizer_config.allow_line_switching and "line_switch" in active_types:
        actions.extend(_line_switching_actions(net, report, graph, optimizer_config))
    if optimizer_config.mode == "ac" and optimizer_config.allow_reactive_load_adjustment and "reactive_load_adjustment" in active_types:
        actions.extend(_reactive_load_actions(net, report, affected, optimizer_config))
    if optimizer_config.allow_load_curtailment and "load_curtailment" in active_types:
        actions.extend(_load_curtailment_actions(net, affected, optimizer_config))
    actions = _filter_tiny_actions(actions, optimizer_config)
    actions = _deduplicate_sort_and_cap(actions, optimizer_config, affected, graph)
    return rank_candidates_fast(actions, report, affected, graph, optimizer_config)[: optimizer_config.max_candidates_total]


def _active_action_types(
    report: ViolationReport,
    config: CorrectiveOptimizerConfig,
    escalation_level: int = 0,
) -> set[str]:
    thermal = report.line_overload_severity > 0.0 or report.trafo_overload_severity > 0.0
    voltage = report.voltage_deviation_severity > 0.0
    if voltage and not thermal and config.mode == "ac":
        active = {
            "gen_voltage_setpoint",
            "ext_grid_voltage_setpoint",
            "trafo_tap_change",
            "reactive_load_adjustment",
        }
        if escalation_level >= 2 or _voltage_problem_is_severe(report):
            active.add("load_curtailment")
        return active
    active = {"gen_redispatch_pair"}
    if config.mode == "ac":
        active.update({"gen_voltage_setpoint", "ext_grid_voltage_setpoint", "trafo_tap_change"})
    if config.allow_line_switching:
        active.add("line_switch")
    if config.allow_load_curtailment:
        active.add("load_curtailment")
    return active


def _voltage_problem_is_severe(report: ViolationReport) -> bool:
    return report.violation_count >= 20 or report.voltage_deviation_severity >= 1.0


def _voltage_problem_kind(report: ViolationReport) -> str:
    if report.low_voltage_count > report.high_voltage_count:
        return "low"
    if report.high_voltage_count > report.low_voltage_count:
        return "high"
    if report.low_voltage_count or report.high_voltage_count:
        return "mixed"
    return "none"


def _directional_voltage_deltas(deltas: tuple[float, ...], report: ViolationReport) -> np.ndarray:
    values = np.asarray(deltas, dtype=float)
    problem = _voltage_problem_kind(report)
    if problem == "low":
        return values[values > 0.0]
    if problem == "high":
        return values[values < 0.0]
    return values


def _directional_reactive_fracs(fracs: tuple[float, ...], report: ViolationReport) -> tuple[float, ...]:
    problem = _voltage_problem_kind(report)
    if problem == "low":
        return tuple(frac for frac in fracs if frac < 0.0)
    if problem == "high":
        return tuple(frac for frac in fracs if frac > 0.0)
    return fracs


def _filter_tiny_actions(actions: list[CorrectiveAction], config: CorrectiveOptimizerConfig) -> list[CorrectiveAction]:
    filtered: list[CorrectiveAction] = []
    removed = Counter()
    for action in actions:
        if action.action_type == "gen_redispatch_pair" and abs(float(action.params.get("delta_mw", 0.0))) < config.min_redispatch_mw:
            removed[action.action_type] += 1
            continue
        if action.action_type == "load_curtailment" and abs(float(action.params.get("curtail_p_mw", 0.0))) < config.min_curtailment_mw:
            removed[action.action_type] += 1
            continue
        if action.action_type == "reactive_load_adjustment" and abs(float(action.params.get("delta_q_mvar", 0.0))) < config.min_reactive_adjustment_mvar:
            removed[action.action_type] += 1
            continue
        if action.action_type in {"gen_voltage_setpoint", "ext_grid_voltage_setpoint"} and abs(float(action.params.get("delta_pu", 0.0))) < config.min_voltage_delta_pu:
            removed[action.action_type] += 1
            continue
        filtered.append(action)
    if removed:
        logger.info("CANDIDATE FILTER tiny_removed=%s", dict(removed))
    return filtered


def _deduplicate_actions(actions: list[CorrectiveAction]) -> list[CorrectiveAction]:
    deduped: dict[tuple, CorrectiveAction] = {}
    for action in actions:
        deduped.setdefault(action.signature(), action)
    return list(deduped.values())


def rank_candidates_fast(
    candidates: list[CorrectiveAction],
    violation_report: ViolationReport,
    affected_buses: set[int],
    graph: dict[int, set[int]] | None = None,
    config: CorrectiveOptimizerConfig | None = None,
    target_adjustment_count: Counter[tuple] | None = None,
) -> list[CorrectiveAction]:
    graph = graph or {}
    config = config or CorrectiveOptimizerConfig()
    target_adjustment_count = target_adjustment_count or Counter()
    voltage_only = (
        violation_report.line_overload_severity == 0.0
        and violation_report.trafo_overload_severity == 0.0
        and violation_report.voltage_deviation_severity > 0.0
    )
    low_voltage = not violation_report.low_voltage_buses.empty
    high_voltage = not violation_report.high_voltage_buses.empty

    def key(action: CorrectiveAction) -> tuple[float, tuple]:
        priority = _fast_action_priority(action, voltage_only)
        direction_penalty = _direction_penalty(action, low_voltage, high_voltage)
        distance_penalty = _action_distance_penalty(action, affected_buses, graph)
        tiny_bonus = -abs(action.magnitude)
        cost_penalty = float(action.cost) + 100.0 * float(action.disruptive_rank)
        repeat_count = target_adjustment_count.get(_action_target_key(action), 0)
        repeat_penalty = 1000.0 * max(0, repeat_count - config.max_repeated_actions_same_target + 1)
        return (priority + direction_penalty + distance_penalty + repeat_penalty + cost_penalty * 0.001 + tiny_bonus * 0.01, action.signature())

    return sorted(candidates, key=key)


def _fast_action_priority(action: CorrectiveAction, voltage_only: bool) -> float:
    if voltage_only:
        priorities = {
            "gen_voltage_setpoint": 0.0,
            "ext_grid_voltage_setpoint": 1.0,
            "trafo_tap_change": 2.0,
            "reactive_load_adjustment": 3.0,
            "gen_redispatch_pair": 20.0,
            "line_switch": 25.0,
            "load_curtailment": 30.0,
        }
    else:
        priorities = {
            "gen_redispatch_pair": 0.0,
            "trafo_tap_change": 2.0,
            "gen_voltage_setpoint": 5.0,
            "ext_grid_voltage_setpoint": 6.0,
            "line_switch": 20.0,
            "reactive_load_adjustment": 25.0,
            "load_curtailment": 30.0,
        }
    return priorities.get(action.action_type, 99.0)


def _direction_penalty(action: CorrectiveAction, low_voltage: bool, high_voltage: bool) -> float:
    if action.action_type in {"gen_voltage_setpoint", "ext_grid_voltage_setpoint"}:
        delta = float(action.params.get("delta_pu", 0.0))
        if low_voltage and delta > 0:
            return -2.0
        if high_voltage and delta < 0:
            return -2.0
        return 3.0
    if action.action_type == "reactive_load_adjustment":
        delta = float(action.params.get("delta_q_mvar", 0.0))
        if low_voltage and delta < 0:
            return -1.5
        if high_voltage and delta > 0:
            return -1.5
        return 2.0
    return 0.0


def _action_distance_penalty(action: CorrectiveAction, affected_buses: set[int], graph: dict[int, set[int]]) -> float:
    if not affected_buses:
        return 0.0
    buses = _action_buses(action)
    if not buses:
        return 1.0
    return min(_graph_distance(graph, bus, affected) for bus in buses for affected in affected_buses) * 0.25


def _action_buses(action: CorrectiveAction) -> list[int]:
    buses = action.params.get("buses")
    if isinstance(buses, (tuple, list)):
        return [int(bus) for bus in buses if _is_finite_number(bus)]
    bus = action.params.get("bus")
    return [int(bus)] if _is_finite_number(bus) else []


def _is_low_cost_action(action: CorrectiveAction) -> bool:
    return action.action_type in {"gen_voltage_setpoint", "ext_grid_voltage_setpoint", "trafo_tap_change"}


def _log_best_candidate_per_type(actions: list[CorrectiveAction]) -> None:
    best: dict[str, CorrectiveAction] = {}
    for action in actions:
        best.setdefault(action.action_type, action)
    if best:
        logger.info("UI BEST CANDIDATE PER TYPE %s", {kind: action.description for kind, action in best.items()})


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
                    params={"delta_mw": round(float(delta), 6), "buses": (int(up.bus), int(down.bus))},
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
    report: ViolationReport,
    affected: set[int],
    config: CorrectiveOptimizerConfig,
) -> list[CorrectiveAction]:
    if not len(net.gen) or "vm_pu" not in net.gen.columns:
        return []
    rows = _local_or_all(_in_service_rows(net.gen), affected)
    deltas = _directional_voltage_deltas(config.action_space.voltage_setpoint_deltas_pu, report)
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
                    {"vm_pu": round(float(new_vm), 6), "delta_pu": float(delta), "bus": bus},
                    cost=abs(delta) * 100.0, disruptive_rank=1,
                    reason=f"Set generator {idx} voltage target to {new_vm:.3f} pu",
                )
            )
    return actions


def _ext_grid_voltage_setpoint_actions(
    net: pp.pandapowerNet,
    report: ViolationReport,
    affected: set[int],
    config: CorrectiveOptimizerConfig,
) -> list[CorrectiveAction]:
    if not len(net.ext_grid) or "vm_pu" not in net.ext_grid.columns:
        return []
    rows = _local_or_all(_in_service_rows(net.ext_grid), affected)
    deltas = _directional_voltage_deltas(config.action_space.ext_grid_voltage_deltas_pu, report)
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
                    {"vm_pu": round(float(new_vm), 6), "delta_pu": float(delta), "bus": bus},
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
                    {"tap_pos": new_tap, "tap_delta": tap_delta, "buses": (int(row.hv_bus), int(row.lv_bus))},
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
                "line_switch", "line", int(idx), {"in_service": not current, "buses": (int(net.line.at[idx, "from_bus"]), int(net.line.at[idx, "to_bus"]))},
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
        for frac in _directional_reactive_fracs(config.action_space.reactive_load_step_fracs, report):
            new_q = current_q + frac * base
            actions.append(
                CorrectiveAction(
                    "reactive_load_adjustment", "load", int(idx),
                    {"q_mvar": round(new_q, 6), "delta_q_mvar": round(new_q - current_q, 6), "bus": int(row.bus)},
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
                        "bus": int(row.bus),
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
    config: CorrectiveOptimizerConfig | None = None,
    current_report: ViolationReport | None = None,
    target_adjustment_count: Counter[tuple] | None = None,
) -> CandidateResult | None:
    seen_states = seen_states or set()
    config = config or CorrectiveOptimizerConfig()
    target_adjustment_count = target_adjustment_count or Counter()
    unseen = [candidate for candidate in candidates if _state_signature(candidate.net) not in seen_states]
    ranked_pool = []
    for candidate in unseen:
        if current_report is not None and not is_acceptable_candidate(
            current_report,
            candidate.violations,
            start_score.score,
            candidate.score.score,
            config,
        ):
            continue
        if current_report is None and not candidate.converged:
            continue
        ranked_pool.append(candidate)
    if not ranked_pool:
        return None
    return min(
        ranked_pool,
        key=lambda candidate: candidate_selection_key(candidate, target_adjustment_count),
    )


def is_acceptable_candidate(
    current_report: ViolationReport,
    candidate_report: ViolationReport,
    current_score: float,
    candidate_score: float,
    config: CorrectiveOptimizerConfig,
    tolerance: float = 1e-6,
) -> bool:
    if not candidate_report.converged:
        return False
    if candidate_report.is_safe:
        return True
    if not config.allow_violation_count_increase and candidate_report.violation_count > current_report.violation_count:
        return False
    if not config.allow_new_voltage_violations and newly_violated_buses(current_report, candidate_report):
        return False
    if candidate_report.violation_count < current_report.violation_count:
        return True
    if candidate_report.max_voltage_deviation > current_report.max_voltage_deviation + tolerance:
        return False
    return candidate_score < current_score - config.min_score_improvement


def candidate_selection_key(candidate: CandidateResult, target_adjustment_count: Counter[tuple] | None = None) -> tuple:
    target_adjustment_count = target_adjustment_count or Counter()
    report = candidate.violations
    action = candidate.action
    repeat_penalty = 1000.0 * target_adjustment_count.get(_action_target_key(action), 0)
    return (
        not report.converged,
        not report.is_safe,
        report.violation_count,
        report.max_voltage_deviation,
        report.voltage_deviation_severity,
        report.line_overload_severity,
        report.trafo_overload_severity,
        float(candidate.score.intervention_cost) + repeat_penalty,
        float(action.disruptive_cost),
    )


def newly_violated_buses(current_report: ViolationReport, candidate_report: ViolationReport) -> set[int]:
    return candidate_report.violated_voltage_buses - current_report.violated_voltage_buses


def resolved_buses(current_report: ViolationReport, candidate_report: ViolationReport) -> set[int]:
    return current_report.violated_voltage_buses - candidate_report.violated_voltage_buses


def worsened_buses(current_report: ViolationReport, candidate_report: ViolationReport) -> set[int]:
    worsened: set[int] = set()
    for bus in current_report.violated_voltage_buses & candidate_report.violated_voltage_buses:
        if _voltage_deviation_for_bus(candidate_report, bus) > _voltage_deviation_for_bus(current_report, bus) + 1e-6:
            worsened.add(bus)
    return worsened


def _voltage_deviation_for_bus(report: ViolationReport, bus: int) -> float:
    total = 0.0
    if bus in report.low_voltage_buses.index:
        total += float(report.low_voltage_buses.at[bus, "low_deviation_pu"])
    if bus in report.high_voltage_buses.index:
        total += float(report.high_voltage_buses.at[bus, "high_deviation_pu"])
    return total


def _action_target_key(action: CorrectiveAction) -> tuple:
    return (action.action_type, _hashable_value(action.target_index))


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
    explanation: str = "",
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
        explanation=explanation,
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


def _rounded_action_param(key: str, value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((item_key, _rounded_action_param(item_key, item_value)) for item_key, item_value in value.items()))
    if isinstance(value, list):
        return tuple(_rounded_action_param(key, item) for item in value)
    if isinstance(value, tuple):
        return tuple(_rounded_action_param(key, item) for item in value)
    if isinstance(value, float):
        if "vm_pu" in key or "delta_pu" in key:
            return round(value, 4)
        if "mw" in key or "mvar" in key:
            return round(value, 3)
        return round(value, 4)
    if key in {"tap_pos", "tap_delta"} and _is_finite_number(value):
        return int(value)
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
