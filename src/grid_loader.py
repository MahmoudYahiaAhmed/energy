from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import pandapower as pp
import pandapower.networks as pn

PowerFlowMode = Literal["dc", "ac"]


@dataclass(frozen=True)
class NetworkOption:
    name: str
    description: str
    factory: Callable[[], pp.pandapowerNet]


NETWORKS: dict[str, NetworkOption] = {
    "IEEE 14-bus": NetworkOption(
        name="IEEE 14-bus",
        description="Small MATPOWER-derived system with generators, lines, loads, and transformers.",
        factory=pn.case14,
    ),
    "IEEE 9-bus": NetworkOption(
        name="IEEE 9-bus",
        description="Very small transmission test case for fast demonstrations.",
        factory=pn.case9,
    ),
    "IEEE 30-bus": NetworkOption(
        name="IEEE 30-bus",
        description="Medium-small test case with more branches and generators than IEEE 14-bus.",
        factory=pn.case30,
    ),
    "IEEE 57-bus": NetworkOption(
        name="IEEE 57-bus",
        description="Medium transmission test case with a richer N-1 contingency space.",
        factory=pn.case57,
    ),
    "IEEE 118-bus": NetworkOption(
        name="IEEE 118-bus",
        description="Larger IEEE test case suited for DC N-1 screening experiments.",
        factory=pn.case118,
    ),
    "IEEE 300-bus": NetworkOption(
        name="IEEE 300-bus",
        description="Large MATPOWER-derived case for more substantial DC screening runs.",
        factory=pn.case300,
    ),
    "case89pegase": NetworkOption(
        name="case89pegase",
        description="Small PEGASE transmission case with 89 buses; useful for faster AC corrective-action experiments.",
        factory=lambda: getattr(pn, "case89pegase")(),
    ),
    "case1354pegase": NetworkOption(
        name="case1354pegase",
        description="Large PEGASE case suitable for testing GridSFM-size guards and pandapower AC validation.",
        factory=lambda: getattr(pn, "case1354pegase")(),
    ),
    "case1888rte": NetworkOption(
        name="case1888rte",
        description="Large RTE case suitable for GridSFM-size screening experiments.",
        factory=lambda: getattr(pn, "case1888rte")(),
    ),
    "case2848rte": NetworkOption(
        name="case2848rte",
        description="Large RTE case suitable for GridSFM-size screening experiments.",
        factory=lambda: getattr(pn, "case2848rte")(),
    ),
    "case6470rte": NetworkOption(
        name="case6470rte",
        description="Very large RTE case for optional large-network experiments.",
        factory=lambda: getattr(pn, "case6470rte")(),
    ),
    "case9241pegase": NetworkOption(
        name="case9241pegase",
        description="Very large PEGASE case for optional large-network experiments.",
        factory=lambda: getattr(pn, "case9241pegase")(),
    ),
}


def available_networks() -> list[str]:
    return list(NETWORKS)


def load_network(name: str = "IEEE 14-bus") -> pp.pandapowerNet:
    option = NETWORKS.get(name) or _network_by_alias(name)
    if option is None:
        raise ValueError(f"Unknown network '{name}'. Choose one of: {', '.join(available_networks())}")

    net = option.factory()
    _ensure_limits(net)
    _ensure_controllability(net)
    return net


def _network_by_alias(name: str) -> NetworkOption | None:
    normalized = name.lower().strip().replace("_", "").replace("-", "").replace(" ", "")
    aliases = {
        "case9": "IEEE 9-bus",
        "ieee9": "IEEE 9-bus",
        "case14": "IEEE 14-bus",
        "ieee14": "IEEE 14-bus",
        "case30": "IEEE 30-bus",
        "ieee30": "IEEE 30-bus",
        "case57": "IEEE 57-bus",
        "ieee57": "IEEE 57-bus",
        "case118": "IEEE 118-bus",
        "ieee118": "IEEE 118-bus",
        "case300": "IEEE 300-bus",
        "ieee300": "IEEE 300-bus",
        "case89pegase": "case89pegase",
        "pegase89": "case89pegase",
        "89pegase": "case89pegase",
        "case1354pegase": "case1354pegase",
        "case1888rte": "case1888rte",
        "case2848rte": "case2848rte",
        "case6470rte": "case6470rte",
        "case9241pegase": "case9241pegase",
    }
    canonical = aliases.get(normalized)
    return NETWORKS.get(canonical) if canonical else None


def run_power_flow(net: pp.pandapowerNet, mode: PowerFlowMode = "dc") -> bool:
    try:
        if mode == "ac":
            pp.runpp(net, calculate_voltage_angles=True, init="dc")
        else:
            pp.rundcpp(net)
    except Exception:
        net.converged = False
        return False
    return bool(getattr(net, "converged", False))


def _ensure_limits(net: pp.pandapowerNet) -> None:
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
    net.bus["min_vm_pu"] = net.bus["min_vm_pu"].fillna(0.95)
    net.bus["max_vm_pu"] = net.bus["max_vm_pu"].fillna(1.05)


def _ensure_controllability(net: pp.pandapowerNet) -> None:
    if len(net.gen):
        if "controllable" not in net.gen.columns:
            net.gen["controllable"] = True
        net.gen["controllable"] = net.gen["controllable"].fillna(True)
        if "min_p_mw" not in net.gen.columns:
            net.gen["min_p_mw"] = 0.0
        if "max_p_mw" not in net.gen.columns:
            net.gen["max_p_mw"] = (net.gen["p_mw"].abs() * 1.4).clip(lower=1.0)
        net.gen["min_p_mw"] = net.gen["min_p_mw"].fillna(0.0)
        net.gen["max_p_mw"] = net.gen["max_p_mw"].fillna((net.gen["p_mw"].abs() * 1.4).clip(lower=1.0))
