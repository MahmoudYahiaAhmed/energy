from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contingency import apply_contingency, list_contingencies
from corrective_agent import run_corrective_agent
from explain import explain_result
from grid_loader import available_networks, load_network, run_power_flow
from violation_detector import ViolationReport, detect_violations
from visualization import bus_voltage_table, generator_table, line_loading_table, network_figure


def _case_view(case_net, report: ViolationReport, title: str, power_flow_mode: str) -> None:
    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(network_figure(case_net, title), use_container_width=True)
    with right:
        st.write("Security status")
        if report.is_safe:
            if power_flow_mode == "ac":
                st.success("No hard thermal or voltage violations.")
            else:
                st.success("No hard thermal violations in the DC power-flow screen.")
        elif not report.converged:
            st.error("Power flow did not converge.")
        else:
            st.error(f"{report.violation_count} violation(s) detected.")
            if len(report.overloaded_lines):
                st.write("Overloaded lines")
                st.dataframe(report.overloaded_lines, use_container_width=True)
            if len(report.low_voltage_buses):
                st.write("Low-voltage buses")
                st.dataframe(report.low_voltage_buses, use_container_width=True)
            if len(report.high_voltage_buses):
                st.write("High-voltage buses")
                st.dataframe(report.high_voltage_buses, use_container_width=True)

    st.write("Bus voltages" if power_flow_mode == "ac" else "Bus voltage angles")
    st.dataframe(bus_voltage_table(case_net), use_container_width=True, hide_index=True)
    st.write("Line loadings")
    st.dataframe(line_loading_table(case_net), use_container_width=True, hide_index=True)
    st.write("Generator outputs")
    gen_table = generator_table(case_net)
    st.dataframe(gen_table if len(gen_table) else pd.DataFrame(), use_container_width=True, hide_index=True)


def _candidate_table(candidates) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "action": candidate.action.description,
                "type": candidate.action.action_type,
                "converged": candidate.converged,
                "safe": candidate.safe,
                "remaining_violations": candidate.score.remaining_violations,
                "overload_amount": round(candidate.score.overload_amount, 3),
                "cost": round(candidate.score.intervention_cost, 2),
                "score": round(candidate.score.score, 2),
            }
        )
    return pd.DataFrame(rows)


def _path_table(steps) -> pd.DataFrame:
    rows = []
    for step in steps:
        chosen = step.chosen
        rows.append(
            {
                "step": step.step_number,
                "action": chosen.action.description,
                "type": chosen.action.action_type,
                "start_score": round(step.start_score.score, 2),
                "end_score": round(chosen.score.score, 2),
                "remaining_violations": chosen.score.remaining_violations,
                "safe": chosen.safe,
            }
        )
    return pd.DataFrame(rows)


STOP_REASON_LABELS = {
    "already_stable": "post-contingency case was already stable",
    "stable": "reached a stable grid state",
    "no_candidates": "no candidate actions were generated",
    "no_converged_candidate": "no candidate action produced a converged power flow",
    "no_improving_candidate": "no candidate improved the score",
    "max_steps_reached": "maximum greedy step count reached",
}


st.set_page_config(page_title="Agentic N-1 Grid Security", layout="wide")

st.title("Agentic N-1 Power-Grid Security Prototype")
st.caption(
    "Offline research prototype. Candidate actions are validated with the selected pandapower power-flow mode "
    "and are not production grid-control recommendations."
)

with st.sidebar:
    st.header("Study setup")
    network_name = st.selectbox("Sample grid", available_networks(), index=0)
    compute_profile = st.selectbox(
        "Compute profile",
        ["Auto", "Balanced", "Fast", "Max speed"],
        index=2,
        help="Fast and Max speed reduce bus-level candidate search and use smaller action sets for quicker results.",
    )
    power_flow_label = st.selectbox(
        "Power-flow model",
        ["DC power flow", "AC power flow"],
        index=0,
    )
    power_flow_mode = "ac" if power_flow_label == "AC power flow" else "dc"
    net = load_network(network_name)
    line_limit = st.slider(
        "Line loading limit (%)",
        min_value=20,
        max_value=150,
        value=100,
        step=5,
        help="Lower this to create stressed study cases where corrective steps are needed.",
    )
    if len(net.line):
        net.line["max_loading_percent"] = float(line_limit)
    contingencies = list_contingencies(net)
    labels = [contingency.label for contingency in contingencies]
    selected_label = st.selectbox("N-1 contingency", labels)
    selected = contingencies[labels.index(selected_label)]

base_ok = run_power_flow(net, mode=power_flow_mode)
base_report = detect_violations(net, include_voltage=power_flow_mode == "ac")
post_net = apply_contingency(net, selected)
post_ok = run_power_flow(post_net, mode=power_flow_mode)
post_report = detect_violations(post_net, include_voltage=power_flow_mode == "ac")
profile_map = {
    "Auto": "auto",
    "Balanced": "balanced",
    "Fast": "fast",
    "Max speed": "max_speed",
}
run_agent_params = inspect.signature(run_corrective_agent).parameters
if "performance_mode" in run_agent_params:
    agent = run_corrective_agent(
        post_net,
        post_report,
        power_flow_mode=power_flow_mode,
        performance_mode=profile_map[compute_profile],
    )
else:
    agent = run_corrective_agent(post_net, post_report, power_flow_mode=power_flow_mode)
final_report = detect_violations(agent.final_net, include_voltage=power_flow_mode == "ac")

summary_cols = st.columns(4)
summary_cols[0].metric("Base PF", "converged" if base_ok else "failed")
summary_cols[1].metric("Post-contingency violations", post_report.violation_count)
summary_cols[2].metric("Greedy steps", len(agent.path), help="Number of accepted corrective actions.")
summary_cols[3].metric("Final status", "SAFE" if final_report.is_safe else "UNSAFE")

st.subheader("Selected contingency")
st.info(selected.label)

tab_base, tab_post, tab_final, tab_agent = st.tabs(
    ["Base case", "After contingency", "After corrective action", "Agent reasoning"]
)

with tab_base:
    _case_view(net, base_report, "Base grid topology", power_flow_mode)

with tab_post:
    _case_view(post_net, post_report, "Post-contingency topology", power_flow_mode)

with tab_final:
    _case_view(agent.final_net, final_report, "Corrected topology", power_flow_mode)

with tab_agent:
    st.subheader("Agent loop")
    st.write("Observe:", agent.observation)
    st.write("Think:", agent.thought)
    st.write("Stop reason:", STOP_REASON_LABELS.get(agent.stop_reason, agent.stop_reason))
    if agent.path:
        st.success(f"Reached final state after {len(agent.path)} greedy step(s).")
        st.subheader("Path to stability")
        st.dataframe(_path_table(agent.path), use_container_width=True, hide_index=True)
        st.write("Final chosen action:", agent.chosen.action.description if agent.chosen else "None")
    else:
        st.warning("No candidate action was selected.")

    st.subheader("Candidate actions")
    st.caption("All candidates evaluated across the greedy search path.")
    st.dataframe(_candidate_table(agent.candidates), use_container_width=True, hide_index=True)

    st.subheader("Explanation")
    st.write(explain_result(selected, post_report, agent, power_flow_mode))

    st.subheader("Limitations")
    st.warning(
        "This is an offline research and education prototype. It omits protection systems, operator procedures, "
        "dynamic stability, uncertainty, market constraints, and real-time telemetry validation."
    )
