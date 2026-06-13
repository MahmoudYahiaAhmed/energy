from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pandapower as pp


@dataclass
class ViolationReport:
    converged: bool
    overloaded_lines: pd.DataFrame = field(default_factory=pd.DataFrame)
    overloaded_trafos: pd.DataFrame = field(default_factory=pd.DataFrame)
    low_voltage_buses: pd.DataFrame = field(default_factory=pd.DataFrame)
    high_voltage_buses: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def violation_count(self) -> int:
        return sum(len(frame) for frame in self.frames())

    @property
    def is_safe(self) -> bool:
        return self.converged and self.violation_count == 0

    def frames(self) -> tuple[pd.DataFrame, ...]:
        return (
            self.overloaded_lines,
            self.overloaded_trafos,
            self.low_voltage_buses,
            self.high_voltage_buses,
        )


def detect_violations(net: pp.pandapowerNet) -> ViolationReport:
    converged = bool(getattr(net, "converged", False))
    if not converged:
        return ViolationReport(converged=False)

    overloaded_lines = _line_violations(net)
    overloaded_trafos = _trafo_violations(net)
    low_voltage_buses, high_voltage_buses = _voltage_violations(net)
    return ViolationReport(
        converged=converged,
        overloaded_lines=overloaded_lines,
        overloaded_trafos=overloaded_trafos,
        low_voltage_buses=low_voltage_buses,
        high_voltage_buses=high_voltage_buses,
    )


def _line_violations(net: pp.pandapowerNet) -> pd.DataFrame:
    if not len(net.line) or not hasattr(net, "res_line"):
        return pd.DataFrame()
    res = net.res_line.join(net.line[["from_bus", "to_bus", "max_loading_percent", "in_service"]])
    res = res[res["in_service"].astype(bool)]
    res["limit_percent"] = res["max_loading_percent"].fillna(100.0)
    res["overload_percent"] = res["loading_percent"] - res["limit_percent"]
    return res[res["overload_percent"] > 0].sort_values("overload_percent", ascending=False)


def _trafo_violations(net: pp.pandapowerNet) -> pd.DataFrame:
    if not len(net.trafo) or not hasattr(net, "res_trafo"):
        return pd.DataFrame()
    res = net.res_trafo.join(net.trafo[["hv_bus", "lv_bus", "max_loading_percent", "in_service"]])
    res = res[res["in_service"].astype(bool)]
    res["limit_percent"] = res["max_loading_percent"].fillna(100.0)
    res["overload_percent"] = res["loading_percent"] - res["limit_percent"]
    return res[res["overload_percent"] > 0].sort_values("overload_percent", ascending=False)


def _voltage_violations(net: pp.pandapowerNet) -> tuple[pd.DataFrame, pd.DataFrame]:
    res = net.res_bus.join(net.bus[["name", "min_vm_pu", "max_vm_pu"]])
    res["low_deviation_pu"] = res["min_vm_pu"] - res["vm_pu"]
    res["high_deviation_pu"] = res["vm_pu"] - res["max_vm_pu"]
    low = res[res["low_deviation_pu"] > 0].sort_values("low_deviation_pu", ascending=False)
    high = res[res["high_deviation_pu"] > 0].sort_values("high_deviation_pu", ascending=False)
    return low, high
