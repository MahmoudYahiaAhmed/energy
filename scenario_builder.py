from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandapower as pp


@dataclass(frozen=True)
class BaseProfiles:
    load_p_mw: pd.Series
    load_q_mvar: pd.Series | None
    sgen_p_mw: pd.Series | None


def capture_base_profiles(net: pp.pandapowerNet) -> BaseProfiles:
    return BaseProfiles(
        load_p_mw=net.load["p_mw"].copy(),
        load_q_mvar=net.load["q_mvar"].copy() if "q_mvar" in net.load.columns else None,
        sgen_p_mw=net.sgen["p_mw"].copy() if len(net.sgen) and "p_mw" in net.sgen.columns else None,
    )


def build_scenarios(smard_df: pd.DataFrame, net: pp.pandapowerNet) -> list[dict]:
    """Convert SMARD system-level data into grid operating scenarios.

    SMARD is German aggregate system data. It does not say which bus consumes or generates power.
    This prototype distributes the time-series signal over the existing pandapower bus/load
    distribution by applying profile multipliers to the base network.
    """
    reference_load = float(smard_df["load_mw"].median())
    if reference_load <= 0:
        raise ValueError("SMARD median load must be positive.")

    renewable_columns = [column for column in ("solar_mw", "wind_mw") if column in smard_df.columns]
    renewable_reference = None
    if renewable_columns:
        renewable_total = smard_df[renewable_columns].sum(axis=1)
        renewable_reference = float(renewable_total.median())

    scenarios: list[dict] = []
    for idx, row in smard_df.iterrows():
        renewable_mw = float(row[renewable_columns].sum()) if renewable_columns else None
        scenarios.append(
            {
                "scenario_id": int(idx),
                "timestamp": row["timestamp"],
                "load_mw": float(row["load_mw"]),
                "solar_mw": float(row["solar_mw"]) if "solar_mw" in row and pd.notna(row["solar_mw"]) else None,
                "wind_mw": float(row["wind_mw"]) if "wind_mw" in row and pd.notna(row["wind_mw"]) else None,
                "load_multiplier": float(row["load_mw"]) / reference_load,
                "renewable_multiplier": (
                    renewable_mw / renewable_reference
                    if renewable_mw is not None and renewable_reference and renewable_reference > 0
                    else None
                ),
            }
        )
    return scenarios


def apply_scenario_to_pandapower(
    net: pp.pandapowerNet,
    scenario: dict,
    base_loads: BaseProfiles,
) -> None:
    """Apply one SMARD-derived scenario to the pandapower network in place."""
    load_multiplier = float(scenario.get("load_multiplier", 1.0))
    net.load["p_mw"] = base_loads.load_p_mw * load_multiplier
    if base_loads.load_q_mvar is not None:
        net.load["q_mvar"] = base_loads.load_q_mvar * load_multiplier

    renewable_multiplier = scenario.get("renewable_multiplier")
    if renewable_multiplier is not None and base_loads.sgen_p_mw is not None and len(net.sgen):
        # This is a coarse distribution. Production studies should map SMARD renewable categories
        # to explicit renewable generators by location and technology.
        net.sgen["p_mw"] = base_loads.sgen_p_mw * float(renewable_multiplier)
