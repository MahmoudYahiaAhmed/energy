from contingency import apply_contingency, list_contingencies
import corrective_agent
from corrective_agent import CorrectiveOptimizerConfig, evaluate_candidate_actions, generate_candidate_actions, run_corrective_agent
from grid_loader import load_network, run_power_flow
from violation_detector import detect_violations


def test_agent_generates_and_evaluates_candidates():
    net = load_network("IEEE 14-bus")
    contingency = next(item for item in list_contingencies(net) if item.kind == "line")
    post = apply_contingency(net, contingency)
    post.line["max_loading_percent"] = 20.0

    assert run_power_flow(post)
    report = detect_violations(post)
    actions = generate_candidate_actions(post, report)
    result = run_corrective_agent(post, report)

    assert actions
    assert result.candidates
    assert result.chosen is not None
    assert result.path
    assert result.path[-1].chosen is result.chosen
    assert result.final_net is not None
    assert result.stop_reason in {
        "stable",
        "no_converged_candidate",
        "no_improving_candidate",
        "max_steps_reached",
        "max_steps_reached_but_improving",
    }


def test_agent_no_action_when_no_controls_needed_still_returns_result():
    net = load_network("IEEE 14-bus")
    assert run_power_flow(net)
    report = detect_violations(net)

    result = run_corrective_agent(net, report)

    assert result.observation
    assert result.final_net is not None
    assert result.path == []
    assert result.stop_reason == "already_stable"


def test_ac_validations_are_capped(monkeypatch):
    net = load_network("IEEE 14-bus")
    contingency = next(item for item in list_contingencies(net) if item.kind == "line")
    post = apply_contingency(net, contingency)
    post.line["max_loading_percent"] = 20.0
    assert run_power_flow(post, mode="dc")
    report = detect_violations(post)
    calls = {"count": 0}

    def counted_run_power_flow(trial, mode="dc"):
        calls["count"] += 1
        return run_power_flow(trial, mode=mode)

    monkeypatch.setattr(corrective_agent, "run_power_flow", counted_run_power_flow)
    config = CorrectiveOptimizerConfig(mode="dc", max_ac_validations_per_step=3, max_candidates_total=40)

    evaluate_candidate_actions(post, report, "dc", config)

    assert calls["count"] <= 3
