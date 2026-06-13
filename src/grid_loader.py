from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandapower as pp
import pandapower.networks as pn


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
}


def available_networks() -> list[str]:
    return list(NETWORKS)


def load_network(name: str = "IEEE 14-bus") -> pp.pandapowerNet:
    option = NETWORKS.get(name)
    if option is None:
        raise ValueError(f"Unknown network '{name}'. Choose one of: {', '.join(available_networks())}")

    net = option.factory()
    _ensure_limits(net)
    _ensure_controllability(net)
    return net


def run_power_flow(net: pp.pandapowerNet) -> bool:
    try:
        pp.runpp(net, calculate_voltage_angles=True, init="auto")
    except Exception:
        try:
            pp.runpp(net, calculate_voltage_angles=False, init="dc")
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
