from copy import deepcopy

from contingency import apply_contingency, list_contingencies
from corrective_agent import (
    ActionCostModel,
    ACPowerFlowValidator,
    CandidateResult,
    CorrectiveAction,
    CorrectiveOptimizerConfig,
    LookaheadPlanner,
    PlannerConfig,
    PlannedSequence,
    ViolationDetector,
    _active_action_types,
    _agent_optimizer_config,
    candidate_selection_key,
    _deduplicate_sort_and_cap,
    _filter_tiny_actions,
    is_acceptable_candidate,
    generate_corrective_candidates,
    newly_violated_buses,
    optimize_post_contingency,
    rank_candidates_fast,
)
from scoring import ScoreBreakdown
from grid_loader import load_network, run_power_flow
from gridsfm_adapter import GridSFMAdapter
from gridsfm_mapper import pandapower_net_to_gridsfm_pyg_json, validate_gridsfm_payload
from violation_detector import ViolationReport, detect_violations


def _stressed_case():
    net = load_network("IEEE 14-bus")
    contingency = next(item for item in list_contingencies(net) if item.kind == "line")
    post = apply_contingency(net, contingency)
    post.line["max_loading_percent"] = 20.0
    assert run_power_flow(post, mode="dc")
    return net, contingency, post, detect_violations(post, include_voltage=False)


def test_redispatch_pair_preserves_total_p_and_respects_bounds():
    _, _, post, report = _stressed_case()
    config = CorrectiveOptimizerConfig(mode="dc", allow_load_curtailment=False)
    actions = generate_corrective_candidates(post, report, config)
    redispatch = next(action for action in actions if action.action_type == "gen_redispatch_pair")

    trial = deepcopy(post)
    before_total = float(trial.gen.p_mw.sum())
    redispatch.apply(trial)
    after_total = float(trial.gen.p_mw.sum())

    assert abs(before_total - after_total) < 1e-6
    up_idx, down_idx = redispatch.target_index
    assert trial.gen.at[up_idx, "p_mw"] <= trial.gen.at[up_idx, "max_p_mw"] + 1e-6
    assert trial.gen.at[down_idx, "p_mw"] >= trial.gen.at[down_idx, "min_p_mw"] - 1e-6


def test_candidate_application_uses_trial_copy_only():
    _, _, post, report = _stressed_case()
    actions = generate_corrective_candidates(post, report, CorrectiveOptimizerConfig(mode="dc"))
    action = actions[0]
    before_signature = tuple(post.gen.p_mw.round(6).tolist())

    trial = deepcopy(post)
    action.apply(trial)

    assert tuple(post.gen.p_mw.round(6).tolist()) == before_signature


def test_candidate_caps_follow_configured_action_type_limits():
    config = CorrectiveOptimizerConfig(
        max_candidates_per_type=100,
        max_gen_redispatch_candidates=3,
        max_gen_voltage_setpoint_candidates=2,
        max_trafo_tap_candidates=2,
        max_ext_grid_voltage_candidates=1,
        max_reactive_load_candidates=2,
        max_load_curtailment_candidates=1,
        max_line_switch_candidates=0,
    )
    actions = [
        CorrectiveAction(action_type, "table", idx, {"value": idx}, 1.0, 1, f"{action_type} {idx}")
        for action_type in (
            "gen_redispatch_pair",
            "gen_voltage_setpoint",
            "trafo_tap_change",
            "ext_grid_voltage_setpoint",
            "reactive_load_adjustment",
            "load_curtailment",
            "line_switch",
        )
        for idx in range(5)
    ]

    capped = _deduplicate_sort_and_cap(actions, config, set(), {})
    counts = {action_type: 0 for action_type in {action.action_type for action in actions}}
    for action in capped:
        counts[action.action_type] += 1

    assert counts["gen_redispatch_pair"] == 3
    assert counts["gen_voltage_setpoint"] == 2
    assert counts["trafo_tap_change"] == 2
    assert counts["ext_grid_voltage_setpoint"] == 1
    assert counts["reactive_load_adjustment"] == 2
    assert counts["load_curtailment"] == 1
    assert counts["line_switch"] == 0


def test_optimizer_forces_pc_mode_even_when_gridsfm_requested():
    net, contingency, _, _ = _stressed_case()
    net.line["max_loading_percent"] = 20.0
    result = optimize_post_contingency(
        net,
        contingency,
        CorrectiveOptimizerConfig(mode="dc", use_gridsfm=True, max_greedy_steps=1),
    )

    assert not result.gridsfm_used
    assert result.gridsfm_skip_reason is not None
    assert "PC/pandapower-only" in result.gridsfm_skip_reason


def test_gridsfm_adapter_rejects_ieee_300_and_smaller_by_bus_count():
    adapter = GridSFMAdapter(checkpoint_path=None, device="cpu", min_buses=500)
    for name in ("case14", "case30", "case57", "case118", "case300"):
        net = load_network(name)
        supported, reason = adapter.is_supported_net(net)
        assert not supported
        assert "below supported minimum 500" in reason


def test_gridsfm_mapper_payload_validates_and_has_reverse_mapping():
    net = load_network("IEEE 14-bus")
    payload = pandapower_net_to_gridsfm_pyg_json(net)
    validate_gridsfm_payload(payload)

    nodes = payload["grid"]["nodes"]
    edges = payload["grid"]["edges"]
    mapping = payload["metadata"]["pandapower_mapping"]

    assert len(nodes["bus"]) == len(net.bus)
    assert "generator_link" in edges
    assert "bus" in mapping
    assert "generator" in mapping
    assert mapping["bus"]


def test_action_cost_model_penalizes_curtailment_more_than_redispatch():
    cost_model = ActionCostModel()
    redispatch = CorrectiveAction("gen_redispatch_pair", "gen", (0, 1), {"delta_mw": 5.0}, 1.0, 1, "redispatch")
    curtailment = CorrectiveAction("load_curtailment", "load", 0, {"curtail_p_mw": 5.0}, 1.0, 1, "curtail")

    assert cost_model.action_cost(curtailment) > cost_model.action_cost(redispatch)


def test_line_switching_is_disabled_by_default():
    _, _, post, report = _stressed_case()
    config = CorrectiveOptimizerConfig(
        mode="dc",
        allow_line_switching=False,
        action_space=CorrectiveOptimizerConfig().action_space.__class__(line_switch_allowlist=(0, 1)),
    )
    actions = generate_corrective_candidates(post, report, config)

    assert all(action.action_type != "line_switch" for action in actions)


def test_lookahead_planner_explores_second_move_and_returns_sequence():
    net = load_network("IEEE 14-bus")
    run_power_flow(net, mode="ac")
    start_report = detect_violations(net, include_voltage=True)
    first = CorrectiveAction("gen_voltage_setpoint", "gen", 0, {"vm_pu": 1.01, "delta_pu": 0.01}, 1.0, 1, "first")
    second = CorrectiveAction("gen_voltage_setpoint", "gen", 1, {"vm_pu": 1.02, "delta_pu": 0.02}, 1.0, 1, "second")

    class FakeGenerator:
        def generate(self, trial, report):
            return [second] if abs(float(trial.gen.at[0, "vm_pu"]) - 1.01) < 1e-9 else [first]

    class FakeValidator(ACPowerFlowValidator):
        def validate_sequence(self, base_net, actions, cost_model):
            report = detect_violations(base_net, include_voltage=True)
            if len(actions) >= 2:
                score = -100.0
            else:
                score = 1_000_000_000.0
            return PlannedSequence(actions, deepcopy(base_net), report, score, score, ac_validated=True)

    class FakeEvaluator:
        def rank(self, seqs):
            return seqs

        def score_trial(self, previous_report, actions, trial, cost_model):
            return float(len(actions))

    planner = LookaheadPlanner(
        PlannerConfig(planning_depth=2, beam_width=2, use_gridsfm=False),
        CorrectiveOptimizerConfig(mode="ac"),
        FakeGenerator(),
        ViolationDetector(),
        FakeEvaluator(),
        FakeValidator(),
        ActionCostModel(),
    )

    sequence, log = planner.plan(net, start_report)

    assert sequence is not None
    assert [action.reason for action in sequence.actions] == ["first", "second"]
    assert log["levels"][1]["expanded"] >= 1


def test_voltage_only_action_types_focus_on_voltage_support():
    net = load_network("IEEE 14-bus")
    assert run_power_flow(net, mode="ac")
    net.bus["max_vm_pu"] = 0.98
    report = detect_violations(net, include_voltage=True)

    active = _active_action_types(report, CorrectiveOptimizerConfig(mode="ac"))

    assert "gen_voltage_setpoint" in active
    assert "ext_grid_voltage_setpoint" in active
    assert "trafo_tap_change" in active
    assert "gen_redispatch_pair" not in active
    assert "load_curtailment" not in active


def test_tiny_actions_are_rejected_by_thresholds():
    config = CorrectiveOptimizerConfig()
    actions = [
        CorrectiveAction("gen_redispatch_pair", "gen", (0, 1), {"delta_mw": 0.40}, 1.0, 1, "tiny redispatch"),
        CorrectiveAction("load_curtailment", "load", 0, {"curtail_p_mw": 0.03}, 1.0, 1, "tiny curtail"),
        CorrectiveAction("reactive_load_adjustment", "load", 0, {"delta_q_mvar": 0.10}, 1.0, 1, "tiny reactive"),
        CorrectiveAction("gen_voltage_setpoint", "gen", 0, {"delta_pu": 0.001, "vm_pu": 1.001}, 1.0, 1, "tiny voltage"),
        CorrectiveAction("gen_redispatch_pair", "gen", (0, 1), {"delta_mw": 5.00}, 1.0, 1, "useful redispatch"),
    ]

    filtered = _filter_tiny_actions(actions, config)

    assert [action.description for action in filtered] == ["useful redispatch"]


def test_candidate_count_is_capped_by_default():
    _, _, post, report = _stressed_case()
    actions = generate_corrective_candidates(post, report, CorrectiveOptimizerConfig(mode="dc"))

    assert len(actions) <= 40


def test_duplicate_voltage_setpoints_are_removed_with_rounded_signature():
    config = CorrectiveOptimizerConfig(max_candidates_total=10, max_candidates_per_type=10)
    actions = [
        CorrectiveAction("gen_voltage_setpoint", "gen", 0, {"vm_pu": 1.01001, "delta_pu": 0.01001}, 1.0, 1, "a"),
        CorrectiveAction("gen_voltage_setpoint", "gen", 0, {"vm_pu": 1.01002, "delta_pu": 0.01002}, 1.0, 1, "b"),
    ]

    capped = _deduplicate_sort_and_cap(actions, config, set(), {})

    assert len(capped) == 1


def test_fast_ranking_prioritizes_correct_voltage_direction():
    net = load_network("IEEE 14-bus")
    assert run_power_flow(net, mode="ac")
    net.bus["min_vm_pu"] = 1.10
    net.bus["max_vm_pu"] = 1.20
    report = detect_violations(net, include_voltage=True)
    up = CorrectiveAction("gen_voltage_setpoint", "gen", 0, {"vm_pu": 1.02, "delta_pu": 0.02, "bus": 0}, 1.0, 1, "up")
    down = CorrectiveAction("gen_voltage_setpoint", "gen", 0, {"vm_pu": 0.98, "delta_pu": -0.02, "bus": 0}, 1.0, 1, "down")

    ranked = rank_candidates_fast([down, up], report, {0}, {0: set()}, CorrectiveOptimizerConfig(mode="ac"))

    assert ranked[0] is up


def test_candidate_that_increases_violation_count_is_rejected():
    net = load_network("IEEE 14-bus")
    assert run_power_flow(net, mode="ac")
    current = detect_violations(net, include_voltage=True)
    net.bus["max_vm_pu"] = 0.98
    worse = detect_violations(net, include_voltage=True)

    assert not is_acceptable_candidate(current, worse, 1000.0, 0.0, CorrectiveOptimizerConfig(mode="ac"))


def test_candidate_that_creates_new_voltage_violation_is_rejected():
    net = load_network("IEEE 14-bus")
    assert run_power_flow(net, mode="ac")
    current = detect_violations(net, include_voltage=True)
    net.bus.at[0, "max_vm_pu"] = 0.95
    candidate = detect_violations(net, include_voltage=True)

    assert newly_violated_buses(current, candidate)
    assert not is_acceptable_candidate(current, candidate, 1000.0, 0.0, CorrectiveOptimizerConfig(mode="ac"))


def test_safe_candidate_beats_unsafe_candidate_with_lower_cost():
    net = load_network("IEEE 14-bus")
    assert run_power_flow(net, mode="ac")
    safe_report = ViolationReport(converged=True)
    unsafe_report = detect_violations(net, include_voltage=True)
    unsafe_action = CorrectiveAction("load_curtailment", "load", 0, {"curtail_p_mw": 1.0}, 1.0, 10, "unsafe")
    safe_action = CorrectiveAction("gen_voltage_setpoint", "gen", 0, {"delta_pu": 0.01, "vm_pu": 1.01}, 100.0, 1, "safe")
    unsafe = CandidateResult(unsafe_action, True, False, ScoreBreakdown(1.0, 1, 0.0, 0.1, 1.0), unsafe_report, net)
    safe = CandidateResult(safe_action, True, True, ScoreBreakdown(100.0, 0, 0.0, 0.0, 100.0), safe_report, net)

    assert candidate_selection_key(safe) < candidate_selection_key(unsafe)


def test_effective_max_steps_respects_configured_max_steps():
    net = load_network("IEEE 14-bus")
    config = _agent_optimizer_config(net, "ac", 12, "fast")

    assert config.max_greedy_steps == 12


def test_repeated_same_target_actions_get_penalized():
    net = load_network("IEEE 14-bus")
    assert run_power_flow(net, mode="ac")
    report = detect_violations(net, include_voltage=True)
    repeated = CorrectiveAction("gen_voltage_setpoint", "gen", 0, {"delta_pu": 0.01, "vm_pu": 1.01, "bus": 0}, 1.0, 1, "repeat")
    fresh = CorrectiveAction("gen_voltage_setpoint", "gen", 1, {"delta_pu": 0.01, "vm_pu": 1.01, "bus": 1}, 1.0, 1, "fresh")

    ranked = rank_candidates_fast(
        [repeated, fresh],
        report,
        {0, 1},
        {0: {1}, 1: {0}},
        CorrectiveOptimizerConfig(mode="ac", max_repeated_actions_same_target=1),
        target_adjustment_count={(repeated.action_type, repeated.target_index): 2},
    )

    assert ranked[0] is fresh
