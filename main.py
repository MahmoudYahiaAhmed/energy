from __future__ import annotations

import time

from config import load_config
from contingency_generator import generate_n1_candidates
from data_loader import load_smard_data
from fallback_screening import dc_screen_contingencies
from gridsfm_oracle import GridSFMOracle
from network_loader import is_gridsfm_suitable, load_network
from pandapower_validator import validate_top_cases
from plan_selector import select_best_plan
from reporting import save_reports
from risk_scoring import rank_candidates
from scenario_builder import build_scenarios


def main() -> None:
    total_start = time.perf_counter()
    config = load_config()

    start = time.perf_counter()
    smard_df = load_smard_data(str(config.smard_csv_path))
    print(f"Loaded SMARD data: {len(smard_df)} timestamps ({time.perf_counter() - start:.2f}s)")

    start = time.perf_counter()
    net = load_network(config.network_name)
    print(f"Loaded network: {len(net.bus)} buses, {len(net.line)} lines ({time.perf_counter() - start:.2f}s)")

    scenarios = build_scenarios(smard_df, net)
    contingencies = generate_n1_candidates(net)
    total_cases = len(scenarios) * len(contingencies)
    print(f"Generated {len(contingencies)} contingencies")
    print(f"Screened total cases: {total_cases}")

    screening_method = _choose_screening_method(config.screening, config.use_gridsfm, config.cpu_only, net)
    print(f"Screening method: {screening_method}")

    start = time.perf_counter()
    if screening_method == "gridsfm":
        oracle = GridSFMOracle(net=net, checkpoint_path=config.gridsfm_checkpoint, device="cpu")
        oracle.load_model()
        predictions = oracle.batch_predict(scenarios, contingencies)
        screened = rank_candidates(predictions, config.top_k)
    elif screening_method == "pandapower":
        screened = dc_screen_contingencies(net, scenarios, contingencies, config.top_k)
        print("Direct pandapower mode requested; still using DC screening to avoid brute-force AC ranking.")
    else:
        screened = dc_screen_contingencies(net, scenarios, contingencies, config.top_k)
    print(f"Selected top {len(screened)} risky cases ({time.perf_counter() - start:.2f}s)")

    start = time.perf_counter()
    validated = validate_top_cases(
        net=net,
        scenarios=scenarios,
        top_cases=screened,
        min_voltage_pu=config.min_voltage_pu,
        max_voltage_pu=config.max_voltage_pu,
        max_line_loading_percent=config.max_line_loading_percent,
        use_lightsim2grid=config.use_lightsim2grid,
    )
    print(f"Validated {len(validated)} cases with pandapower AC ({time.perf_counter() - start:.2f}s)")

    best_plan = select_best_plan(validated)
    avoided = max(0, total_cases - len(validated))
    print(f"Avoided {avoided} full AC runs")

    if not best_plan.empty:
        worst = screened.iloc[0]
        best = best_plan.iloc[0]
        print(f"Worst contingency: {worst['contingency_id']} at {worst['timestamp']}")
        print(f"Best secure plan: {best['contingency_id']} secure={best['secure']}")

    save_reports(screened, validated, best_plan, config.output_folder)
    print(f"Reports saved to {config.output_folder}/")
    print(f"Total runtime: {time.perf_counter() - total_start:.2f}s")


def _choose_screening_method(requested: str, use_gridsfm: bool, cpu_only: bool, net) -> str:
    if cpu_only and requested in {"auto", "gridsfm"}:
        if requested == "gridsfm":
            print("WARNING: GridSFM requested, but --enable-gridsfm was not set. Using DC fallback screening.")
        return "dc"

    suitable = is_gridsfm_suitable(net)
    if requested == "gridsfm" and not suitable:
        print(
            "WARNING: GridSFM requested, but this grid is too small for GridSFM-Open "
            f"({len(net.bus)} buses < 500). Using DC fallback screening."
        )
        return "dc"
    if requested == "auto":
        if use_gridsfm and suitable:
            return "gridsfm"
        print(
            "WARNING: Grid is too small for GridSFM-Open or GridSFM disabled. "
            "Using DC fallback screening; pandapower remains the final validator."
        )
        return "dc"
    return requested


if __name__ == "__main__":
    main()
