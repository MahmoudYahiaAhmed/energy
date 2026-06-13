from __future__ import annotations

from pathlib import Path

import pandapower as pp
import pandapower.networks as pn


GRIDSFM_MIN_BUSES = 500


def load_network(name: str) -> pp.pandapowerNet:
    """Load a pandapower network by name or from a pandapower JSON file."""
    normalized = name.lower().strip()
    if normalized in {"case14", "ieee14", "ieee 14-bus"}:
        net = pn.case14()
    elif normalized in {"case30", "ieee30", "ieee 30-bus"}:
        net = pn.case30()
    elif normalized in {"case57", "ieee57", "ieee 57-bus"}:
        net = pn.case57()
    elif normalized in {"case118", "ieee118", "ieee 118-bus"}:
        net = pn.case118()
    elif normalized in {"case1354pegase", "pegase1354", "case1354"}:
        net = pn.case1354pegase()
    elif normalized in {"case2869pegase", "pegase2869", "case2869"}:
        net = pn.case2869pegase()
    elif normalized in {"case6470rte", "rte6470", "case6470"}:
        net = pn.case6470rte()
    elif normalized in {"case9241pegase", "pegase9241", "case9241"}:
        net = pn.case9241pegase()
    elif normalized.endswith(".json"):
        path = Path(name)
        if not path.exists():
            raise FileNotFoundError(f"Pandapower JSON not found: {path}")
        net = pp.from_json(str(path))
    elif normalized in {"pypsa-eur", "europe-large", "large-eu"}:
        raise NotImplementedError(
            "Large European grid support is a placeholder. Convert PyPSA-Eur or another model "
            "to pandapower JSON and pass that JSON path."
        )
    else:
        raise ValueError(f"Unsupported network '{name}'. Use case14, case30, case57, case118, or a JSON path.")

    _ensure_limits(net)
    return net


def is_gridsfm_suitable(net: pp.pandapowerNet) -> bool:
    """GridSFM-Open is intended for larger grids; small IEEE cases use fallback screening."""
    return len(net.bus) >= GRIDSFM_MIN_BUSES


def _ensure_limits(net: pp.pandapowerNet) -> None:
    if len(net.line):
        if "max_loading_percent" not in net.line.columns:
            net.line["max_loading_percent"] = 100.0
        net.line["max_loading_percent"] = net.line["max_loading_percent"].fillna(100.0)
    if len(net.trafo):
        if "max_loading_percent" not in net.trafo.columns:
            net.trafo["max_loading_percent"] = 100.0
        net.trafo["max_loading_percent"] = net.trafo["max_loading_percent"].fillna(100.0)
    if "min_vm_pu" not in net.bus.columns:
        net.bus["min_vm_pu"] = 0.95
    if "max_vm_pu" not in net.bus.columns:
        net.bus["max_vm_pu"] = 1.05
