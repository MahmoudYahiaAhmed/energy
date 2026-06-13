from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contingency import apply_contingency, list_contingencies
from corrective_agent import run_corrective_agent
from contingency_generator import generate_n1_candidates
from data_loader import load_smard_data as load_pipeline_smard_data
from explain import explain_result
from fallback_screening import dc_screen_contingencies
from gridsfm_oracle import GridSFMOracle
from grid_loader import available_networks, load_network, run_power_flow
from network_loader import is_gridsfm_suitable, load_network as load_pipeline_network
from pandapower_validator import validate_top_cases
from plan_selector import select_best_plan
from scenario_builder import build_scenarios
from smard_loader import apply_smard_snapshot, read_bundled_smard_data, read_smard_csv, snapshot_from_row
from violation_detector import ViolationReport, detect_violations
from visualization import bus_voltage_table, generator_table, line_loading_table, network_figure


def _case_view(case_net, report: ViolationReport, title: str) -> None:
    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(network_figure(case_net, title), use_container_width=True)
    with right:
        st.write("Security status")
        if report.is_safe:
            st.success("No hard thermal or voltage violations.")
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

    st.write("Bus voltages")
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
                "voltage_deviation": round(candidate.score.voltage_deviation, 4),
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
    "max_steps_reached": "maximum greedy step count reached",
}


def _hybrid_platform() -> None:
    st.subheader("Hybrid SMARD Screening Platform")
    st.caption(
        "GridSFM mode uses the real gridsfm package on CPU when installed and configured. "
        "DC screening remains available as a fallback."
    )

    st.markdown(
        "`SMARD scenarios` -> `pandapower/GridSFM graph` -> `GridSFM or DC screener` -> "
        "`top-K risky cases` -> `pandapower AC validator` -> `plan ranking`"
    )

    control_cols = st.columns(5)
    with control_cols[0]:
        pipeline_network = st.selectbox(
            "Pipeline network",
            ["case6470rte", "case2869pegase", "case1354pegase", "case118", "case57", "case30", "case14"],
            index=2,
            key="pipeline_network",
        )
    with control_cols[1]:
        max_timestamps = st.number_input(
            "Timestamps",
            min_value=1,
            max_value=96,
            value=2,
            step=1,
            help="Limit timestamps for interactive CPU runs.",
        )
    with control_cols[2]:
        top_k = st.number_input("Top-K AC validations", min_value=1, max_value=200, value=10, step=1)
    with control_cols[3]:
        max_loading = st.number_input("Line limit (%)", min_value=50.0, max_value=200.0, value=100.0, step=5.0)
    with control_cols[4]:
        screening_method = st.selectbox("Screener", ["GridSFM", "DC fallback"], index=0)

    max_contingencies = st.number_input(
        "Max contingencies to screen",
        min_value=1,
        max_value=5000,
        value=100,
        step=25,
        help="Interactive safety cap. Increase for larger offline runs.",
    )
    smard_path = st.text_input("SMARD CSV path", value="smard_load.csv")
    checkpoint_path = st.text_input(
        "GridSFM checkpoint path",
        value="",
        help="Optional path to gridsfm_open_v1.1.pt. If empty, gridsfm.load_from_hf is used.",
    )
    st.info("GridSFM inference is forced to CPU in this dashboard. Pandapower remains the final validator.")

    if not st.button("Run CPU Screening Pipeline", type="primary"):
        return

    try:
        with st.spinner("Loading SMARD data and network..."):
            smard_df = load_pipeline_smard_data(smard_path).head(int(max_timestamps))
            pipeline_net = load_pipeline_network(pipeline_network)
            if len(pipeline_net.line):
                pipeline_net.line["max_loading_percent"] = float(max_loading)
            scenarios = build_scenarios(smard_df, pipeline_net)
            contingencies = generate_n1_candidates(pipeline_net)[: int(max_contingencies)]
            total_cases = len(scenarios) * len(contingencies)

        if screening_method == "GridSFM" and not is_gridsfm_suitable(pipeline_net):
            st.warning(
                f"{pipeline_network} has {len(pipeline_net.bus)} buses, below the 500-bus GridSFM suitability "
                "threshold. Select a large network such as case6470rte for meaningful GridSFM screening."
            )

        if screening_method == "GridSFM":
            with st.spinner("Converting cases and screening with GridSFM on CPU..."):
                oracle = GridSFMOracle(
                    net=pipeline_net,
                    checkpoint_path=checkpoint_path or None,
                    device="cpu",
                )
                oracle.load_model()
                screened = oracle.batch_predict(scenarios, contingencies)
                screened = screened.sort_values("risk_score", ascending=False).head(int(top_k)).reset_index(drop=True)
        else:
            with st.spinner("Screening all cases with CPU DC power flow..."):
                screened = dc_screen_contingencies(pipeline_net, scenarios, contingencies, int(top_k))

        with st.spinner("Validating top-K cases with pandapower AC..."):
            validated = validate_top_cases(
                net=pipeline_net,
                scenarios=scenarios,
                top_cases=screened,
                max_line_loading_percent=float(max_loading),
                use_lightsim2grid=False,
            )
            best_plan = select_best_plan(validated)

        avoided_runs = max(0, total_cases - len(validated))
        metric_cols = st.columns(5)
        metric_cols[0].metric("SMARD timestamps", len(smard_df))
        metric_cols[1].metric("N-1 contingencies", len(contingencies))
        metric_cols[2].metric("Screened cases", total_cases)
        metric_cols[3].metric("AC validations", len(validated))
        metric_cols[4].metric("AC runs avoided", avoided_runs)

        st.subheader("SMARD Load Profile")
        load_chart = smard_df[["timestamp", "load_mw"]].set_index("timestamp")
        st.line_chart(load_chart)

        st.subheader("Top Risky Cases From CPU Screening")
        st.dataframe(screened, use_container_width=True, hide_index=True)
        if not screened.empty:
            risk_chart = screened[["contingency_id", "risk_score"]].head(20).set_index("contingency_id")
            st.bar_chart(risk_chart)

        st.subheader("Pandapower AC Validation Results")
        st.dataframe(validated, use_container_width=True, hide_index=True)
        if not validated.empty:
            validation_chart = validated[
                ["contingency_id", "max_line_loading_percent", "min_voltage_pu"]
            ].set_index("contingency_id")
            st.line_chart(validation_chart)

        st.subheader("Best Plan Ranking")
        st.dataframe(best_plan, use_container_width=True, hide_index=True)
        if not best_plan.empty:
            best = best_plan.iloc[0]
            st.success(
                f"Best ranked case: {best['contingency_id']} at {best['timestamp']} "
                f"(secure={best['secure']})."
            )
    except Exception as exc:
        st.error(f"Hybrid pipeline failed: {exc}")


st.set_page_config(page_title="Agentic N-1 Grid Security", layout="wide")

st.title("Agentic N-1 Power-Grid Security Prototype")
st.caption(
    "Offline research prototype. Candidate actions are always validated with pandapower AC power flow "
    "and are not production grid-control recommendations."
)

with st.sidebar:
    st.header("Study setup")
    network_name = st.selectbox("Sample grid", available_networks(), index=0)
    model_name = st.selectbox(
        "Model",
        ["pandapower AC power flow", "GridSFM screening in Hybrid tab"],
        index=0,
    )
    if model_name != "pandapower AC power flow":
        st.info("Use the Hybrid SMARD screening tab for GridSFM CPU inference and top-K validation.")
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
    smard_source = st.selectbox(
        "SMARD data",
        ["Bundled SMARD.de sample", "Upload SMARD CSV", "Do not apply SMARD data"],
        index=0,
    )
    smard_file = None
    if smard_source == "Upload SMARD CSV":
        smard_file = st.file_uploader(
            "SMARD CSV snapshot data",
            type=["csv"],
            help="Upload a CSV export from SMARD.de. The selected row scales the pandapower load/generation.",
        )
    if smard_source != "Do not apply SMARD data":
        try:
            if smard_source == "Bundled SMARD.de sample":
                smard_data = read_bundled_smard_data()
            elif smard_file is not None:
                smard_data = read_smard_csv(smard_file.getvalue())
            else:
                smard_data = pd.DataFrame()
            if smard_data.empty:
                st.warning("No usable SMARD load or generation columns were found in this file.")
            else:
                snapshot_labels = [
                    f"{idx}: {row['timestamp']} | load={row['load_mw'] if pd.notna(row['load_mw']) else 'n/a'} MW"
                    f" | multiplier={row['load_multiplier']:.3f}"
                    for idx, row in smard_data.iterrows()
                ]
                snapshot_label = st.selectbox("SMARD timestamp", snapshot_labels, index=0)
                snapshot_idx = int(snapshot_label.split(":", 1)[0])
                snapshot = snapshot_from_row(smard_data.iloc[snapshot_idx])
                apply_smard_snapshot(net, snapshot)
                st.caption(
                    f"Applied SMARD snapshot {snapshot.timestamp}: "
                    f"load={snapshot.load_mw if snapshot.load_mw is not None else 'n/a'} MW, "
                    f"profile multiplier={snapshot.load_multiplier if snapshot.load_multiplier is not None else 'n/a'}."
                )
        except Exception as exc:
            st.error(f"Could not read SMARD CSV: {exc}")
    contingencies = list_contingencies(net)
    labels = [contingency.label for contingency in contingencies]
    selected_label = st.selectbox("N-1 contingency", labels)
    selected = contingencies[labels.index(selected_label)]
    st.divider()
    st.write("Hybrid screening")
    st.caption("GridSFM CPU screening is available in the Hybrid SMARD screening tab.")

base_ok = run_power_flow(net)
base_report = detect_violations(net)
post_net = apply_contingency(net, selected)
post_ok = run_power_flow(post_net)
post_report = detect_violations(post_net)
agent = run_corrective_agent(post_net, post_report)
final_report = detect_violations(agent.final_net)

summary_cols = st.columns(4)
summary_cols[0].metric("Base PF", "converged" if base_ok else "failed")
summary_cols[1].metric("Post-contingency violations", post_report.violation_count)
summary_cols[2].metric("Greedy steps", len(agent.path), help="Number of accepted corrective actions.")
summary_cols[3].metric("Final status", "SAFE" if final_report.is_safe else "UNSAFE")

st.subheader("Selected contingency")
st.info(selected.label)

tab_base, tab_post, tab_final, tab_agent, tab_hybrid = st.tabs(
    ["Base case", "After contingency", "After corrective action", "Agent reasoning", "Hybrid SMARD screening"]
)

with tab_base:
    _case_view(net, base_report, "Base grid topology")

with tab_post:
    _case_view(post_net, post_report, "Post-contingency topology")

with tab_final:
    _case_view(agent.final_net, final_report, "Corrected topology")

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
    st.write(explain_result(selected, post_report, agent))

    st.subheader("Limitations")
    st.warning(
        "This is an offline research and education prototype. It omits protection systems, operator procedures, "
        "dynamic stability, uncertainty, market constraints, and real-time telemetry validation."
    )

with tab_hybrid:
    _hybrid_platform()
