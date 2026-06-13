from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pandapower as pp

from risk_scoring import compute_risk_score
from scenario_builder import BaseProfiles, apply_scenario_to_pandapower, capture_base_profiles


@dataclass
class GridSFMOracle:
    """CPU GridSFM screener for pandapower scenario/contingency cases.

    This adapter uses the real `gridsfm` package when installed. It converts the current
    pandapower network state into GridSFM's native `.pyg.json` layout, calls GridSFM inference,
    and normalizes the output into the same schema as the DC fallback screener.
    """

    net: pp.pandapowerNet
    checkpoint_path: str | Path | None = None
    device: str = "cpu"
    model: Any | None = None
    base_profiles: BaseProfiles | None = None

    def load_model(self) -> None:
        try:
            from gridsfm import load_from_hf, load_model
        except Exception as exc:
            raise RuntimeError(
                "The `gridsfm` package is not installed. Install Microsoft GridSFM model package first:\n"
                "  git clone https://github.com/microsoft/GridSFM.git\n"
                "  cd GridSFM/model\n"
                "  python -m pip install -e .\n"
                "Then pass --enable-gridsfm and optionally --gridsfm-checkpoint path/to/gridsfm_open_v1.1.pt."
            ) from exc

        checkpoint = self.checkpoint_path or os.getenv("GRIDSFM_CHECKPOINT")
        if checkpoint:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"GridSFM checkpoint not found: {checkpoint_path}")
            self.model = load_model(str(checkpoint_path), device=self.device)
        else:
            self.model = load_from_hf("microsoft/GridSFM_Open")
            self.model.to(self.device)
        self.model.eval()
        self.base_profiles = capture_base_profiles(self.net)

    def convert_case_to_gridsfm_input(self, scenario: dict, contingency: dict) -> dict:
        if self.base_profiles is None:
            self.base_profiles = capture_base_profiles(self.net)
        apply_scenario_to_pandapower(self.net, scenario, self.base_profiles)
        original_status = _set_contingency_status(self.net, contingency, False)
        try:
            return pandapower_to_gridsfm_pyg_json(self.net)
        finally:
            _restore_contingency_status(self.net, contingency, original_status)

    def predict_case(self, scenario: dict, contingency: dict) -> dict:
        if self.model is None:
            raise RuntimeError("GridSFM model is not loaded. Call load_model() first.")
        from gridsfm import predict

        pyg_case = self.convert_case_to_gridsfm_input(scenario, contingency)
        edge_rates = pyg_case.pop("_edge_rates_mva")
        with tempfile.NamedTemporaryFile("w", suffix=".pyg.json", delete=False) as handle:
            json.dump(pyg_case, handle)
            temp_path = Path(handle.name)

        try:
            out = predict(self.model, temp_path, fmt="pyg")
        finally:
            temp_path.unlink(missing_ok=True)

        voltage = out["V"]
        pij = out["Pij"]
        flow_mva = [abs(float(value)) * float(self.net.sn_mva) for value in pij]
        loadings = [
            flow / rate * 100.0
            for flow, rate in zip(flow_mva, edge_rates)
            if rate is not None and rate > 0
        ]
        prediction = {
            "predicted_feasible": float(out["feas"]) >= 0.5,
            "predicted_max_line_loading_percent": max(loadings) if loadings else None,
            "predicted_min_voltage_pu": float(voltage.min()) if len(voltage) else None,
            "predicted_max_voltage_pu": float(voltage.max()) if len(voltage) else None,
            "predicted_branch_flows": [float(value) for value in pij],
        }
        prediction["risk_score"] = compute_risk_score(prediction)
        return prediction

    def batch_predict(self, scenarios: list[dict], contingencies: list[dict]) -> pd.DataFrame:
        rows = []
        for scenario in scenarios:
            for contingency in contingencies:
                prediction = self.predict_case(scenario, contingency)
                rows.append(
                    {
                        "scenario_id": scenario["scenario_id"],
                        "timestamp": scenario["timestamp"],
                        "contingency_id": contingency["contingency_id"],
                        "contingency_type": contingency["type"],
                        "element_id": contingency["element_id"],
                        **prediction,
                    }
                )
        return pd.DataFrame(rows)


def pandapower_to_gridsfm_pyg_json(net: pp.pandapowerNet) -> dict:
    """Convert a pandapower network into GridSFM native `.pyg.json` schema.

    The schema follows GridSFM's documented columns:
      bus: [base_kV, type, Vmin, Vmax]
      generator: [mbase, _, Pmin, Pmax, _, Qmin, Qmax, Vg, cp2, cp1, cp0]
      load: [Pd, Qd]
      ac_line: [angmin, angmax, b_fr, b_to, r, x, rate_a, _, _]
      transformer: [angmin, angmax, r, x, rate_a, _, _, tap, shift, b_fr, b_to]
    """
    bus_lookup = {int(bus_id): pos for pos, bus_id in enumerate(net.bus.index)}
    base_mva = float(getattr(net, "sn_mva", 100.0) or 100.0)

    buses = []
    for idx, row in net.bus.iterrows():
        buses.append(
            [
                _float(row.get("vn_kv", 1.0)),
                _bus_type(net, int(idx)),
                _float(row.get("min_vm_pu", 0.95)),
                _float(row.get("max_vm_pu", 1.05)),
            ]
        )

    generators = []
    generator_senders = []
    generator_receivers = []
    if len(net.ext_grid):
        for idx, row in net.ext_grid.iterrows():
            if not bool(row.get("in_service", True)):
                continue
            generators.append([base_mva, 0.0, 0.0, 10.0, 0.0, -10.0, 10.0, _float(row.get("vm_pu", 1.0)), 0.0, 1.0, 0.0])
            generator_senders.append(len(generators) - 1)
            generator_receivers.append(bus_lookup[int(row.bus)])
    for idx, row in net.gen.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        p_mw = _float(row.get("p_mw", 0.0))
        generators.append(
            [
                base_mva,
                0.0,
                _float(row.get("min_p_mw", 0.0)) / base_mva,
                _float(row.get("max_p_mw", max(abs(p_mw), 1.0))) / base_mva,
                0.0,
                _float(row.get("min_q_mvar", -base_mva)) / base_mva,
                _float(row.get("max_q_mvar", base_mva)) / base_mva,
                _float(row.get("vm_pu", 1.0)),
                0.0,
                1.0,
                0.0,
            ]
        )
        generator_senders.append(len(generators) - 1)
        generator_receivers.append(bus_lookup[int(row.bus)])

    loads = []
    load_senders = []
    load_receivers = []
    for idx, row in net.load.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        loads.append([_float(row.get("p_mw", 0.0)) / base_mva, _float(row.get("q_mvar", 0.0)) / base_mva])
        load_senders.append(len(loads) - 1)
        load_receivers.append(bus_lookup[int(row.bus)])

    line_senders = []
    line_receivers = []
    line_features = []
    edge_rates = []
    for idx, row in net.line.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        vn_kv = _float(net.bus.at[int(row.from_bus), "vn_kv"], 1.0)
        z_base = max(vn_kv * vn_kv / base_mva, 1e-9)
        length = _float(row.get("length_km", 1.0), 1.0)
        r = _float(row.get("r_ohm_per_km", 0.0)) * length / z_base
        x = max(_float(row.get("x_ohm_per_km", 1e-4)) * length / z_base, 1e-4)
        b = _float(row.get("c_nf_per_km", 0.0)) * length * 1e-9 * 2.0 * 3.141592653589793 * 50.0 * z_base
        rate = _line_rate_mva(net, int(idx), row)
        line_senders.append(bus_lookup[int(row.from_bus)])
        line_receivers.append(bus_lookup[int(row.to_bus)])
        line_features.append([-0.6, 0.6, b / 2.0, b / 2.0, r, x, rate / base_mva, 0.0, 0.0])
        edge_rates.append(rate)

    trafo_senders = []
    trafo_receivers = []
    trafo_features = []
    for idx, row in net.trafo.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        vk = _float(row.get("vk_percent", 10.0)) / 100.0
        vkr = _float(row.get("vkr_percent", 0.5)) / 100.0
        x = max((max(vk * vk - vkr * vkr, 0.0)) ** 0.5, 1e-4)
        rate = _float(row.get("sn_mva", base_mva), base_mva)
        tap = _float(row.get("tap_pos", 0.0), 0.0)
        trafo_senders.append(bus_lookup[int(row.hv_bus)])
        trafo_receivers.append(bus_lookup[int(row.lv_bus)])
        trafo_features.append([-0.6, 0.6, vkr, x, rate / base_mva, 0.0, 0.0, tap, 0.0, 0.0, 0.0])
        edge_rates.append(rate)

    return {
        "metadata": {"source": "pandapower", "base_mva": base_mva},
        "grid": {
            "nodes": {"bus": buses, "generator": generators, "load": loads, "shunt": []},
            "edges": {
                "ac_line": {"senders": line_senders, "receivers": line_receivers, "features": line_features},
                "transformer": {"senders": trafo_senders, "receivers": trafo_receivers, "features": trafo_features},
                "generator_link": {"senders": generator_senders, "receivers": generator_receivers},
                "load_link": {"senders": load_senders, "receivers": load_receivers},
            },
        },
        "_edge_rates_mva": edge_rates,
    }


def _set_contingency_status(net: pp.pandapowerNet, contingency: dict, in_service: bool) -> bool:
    table = _table_for_type(net, contingency["type"])
    element_id = int(contingency["element_id"])
    original = bool(table.at[element_id, "in_service"]) if "in_service" in table.columns else True
    table.at[element_id, "in_service"] = in_service
    return original


def _restore_contingency_status(net: pp.pandapowerNet, contingency: dict, original_status: bool) -> None:
    table = _table_for_type(net, contingency["type"])
    table.at[int(contingency["element_id"]), "in_service"] = original_status


def _table_for_type(net: pp.pandapowerNet, contingency_type: str):
    if contingency_type == "line":
        return net.line
    if contingency_type == "trafo":
        return net.trafo
    if contingency_type == "gen":
        return net.gen
    raise ValueError(f"Unsupported contingency type: {contingency_type}")


def _bus_type(net: pp.pandapowerNet, bus: int) -> float:
    if len(net.ext_grid) and bus in set(net.ext_grid["bus"].astype(int)):
        return 3.0
    if len(net.gen):
        gen_rows = net.gen
        if "in_service" in gen_rows.columns:
            gen_rows = gen_rows[gen_rows["in_service"].astype(bool)]
        if bus in set(gen_rows["bus"].astype(int)):
            return 2.0
    return 1.0


def _line_rate_mva(net: pp.pandapowerNet, line_idx: int, row: pd.Series) -> float:
    if "max_i_ka" in row and pd.notna(row["max_i_ka"]):
        vn_kv = _float(net.bus.at[int(row.from_bus), "vn_kv"], 1.0)
        return max(3.0 ** 0.5 * vn_kv * _float(row["max_i_ka"], 1.0), 1e-3)
    return max(_float(row.get("max_loading_percent", 100.0), 100.0), 1.0)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default
