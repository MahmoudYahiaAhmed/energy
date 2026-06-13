from copy import deepcopy

from contingency import apply_contingency, list_contingencies
from corrective_agent import (
    CorrectiveAction,
    CorrectiveOptimizerConfig,
    _deduplicate_sort_and_cap,
    generate_corrective_candidates,
    optimize_post_contingency,
)
from grid_loader import load_network, run_power_flow
from gridsfm_adapter import GridSFMAdapter
from gridsfm_mapper import pandapower_net_to_gridsfm_pyg_json, validate_gridsfm_payload
from violation_detector import detect_violations


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


def test_small_ieee_case_skips_gridsfm_guard():
    net, contingency, _, _ = _stressed_case()
    net.line["max_loading_percent"] = 20.0
    result = optimize_post_contingency(
        net,
        contingency,
        CorrectiveOptimizerConfig(mode="dc", use_gridsfm=True, max_greedy_steps=1),
    )

    assert not result.gridsfm_used
    assert result.gridsfm_skip_reason is not None
    assert "below supported minimum 500" in result.gridsfm_skip_reason


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
