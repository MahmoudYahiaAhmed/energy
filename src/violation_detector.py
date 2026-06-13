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
    def low_voltage_count(self) -> int:
        return len(self.low_voltage_buses)

    @property
    def high_voltage_count(self) -> int:
        return len(self.high_voltage_buses)

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

    @property
    def line_overload_severity(self) -> float:
        if self.overloaded_lines.empty:
            return 0.0
        return float(self.overloaded_lines["overload_percent"].clip(lower=0.0).sum())

    @property
    def trafo_overload_severity(self) -> float:
        if self.overloaded_trafos.empty:
            return 0.0
        return float(self.overloaded_trafos["overload_percent"].clip(lower=0.0).sum())

    @property
    def voltage_deviation_severity(self) -> float:
        total = 0.0
        if not self.low_voltage_buses.empty:
            total += float(self.low_voltage_buses["low_deviation_pu"].clip(lower=0.0).sum())
        if not self.high_voltage_buses.empty:
            total += float(self.high_voltage_buses["high_deviation_pu"].clip(lower=0.0).sum())
        return total

    @property
    def low_voltage_severity(self) -> float:
        if self.low_voltage_buses.empty:
            return 0.0
        return float(self.low_voltage_buses["low_deviation_pu"].clip(lower=0.0).sum())

    @property
    def high_voltage_severity(self) -> float:
        if self.high_voltage_buses.empty:
            return 0.0
        return float(self.high_voltage_buses["high_deviation_pu"].clip(lower=0.0).sum())

    @property
    def max_low_voltage_deviation(self) -> float:
        if self.low_voltage_buses.empty:
            return 0.0
        return float(self.low_voltage_buses["low_deviation_pu"].clip(lower=0.0).max())

    @property
    def max_high_voltage_deviation(self) -> float:
        if self.high_voltage_buses.empty:
            return 0.0
        return float(self.high_voltage_buses["high_deviation_pu"].clip(lower=0.0).max())

    @property
    def max_voltage_deviation(self) -> float:
        return max(self.max_low_voltage_deviation, self.max_high_voltage_deviation)

    @property
    def violated_voltage_buses(self) -> set[int]:
        return set(int(idx) for idx in self.low_voltage_buses.index).union(
            int(idx) for idx in self.high_voltage_buses.index
        )

    def to_summary(self) -> dict[str, float | int | bool]:
        return {
            "converged": self.converged,
            "violation_count": self.violation_count,
            "line_overload_severity": self.line_overload_severity,
            "trafo_overload_severity": self.trafo_overload_severity,
            "voltage_deviation_severity": self.voltage_deviation_severity,
            "low_voltage_count": self.low_voltage_count,
            "high_voltage_count": self.high_voltage_count,
            "max_voltage_deviation": self.max_voltage_deviation,
        }


def detect_violations(net: pp.pandapowerNet, include_voltage: bool = False) -> ViolationReport:
    converged = bool(getattr(net, "converged", False))
    if not converged:
        return ViolationReport(converged=False)

    overloaded_lines = _line_violations(net)
    overloaded_trafos = _trafo_violations(net)
    low_voltage_buses, high_voltage_buses = (
        _voltage_violations(net) if include_voltage else (pd.DataFrame(), pd.DataFrame())
    )
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
    line = net.line.copy()
    if "max_loading_percent" not in line.columns:
        line["max_loading_percent"] = 100.0
    if "in_service" not in line.columns:
        line["in_service"] = True
    res = net.res_line.join(line[["from_bus", "to_bus", "max_loading_percent", "in_service"]])
    res = res[res["in_service"].astype(bool)]
    res["limit_percent"] = res["max_loading_percent"].fillna(100.0)
    if "loading_percent" not in res.columns:
        res["loading_percent"] = res["p_from_mw"].abs() if "p_from_mw" in res.columns else 0.0
    elif "p_from_mw" in res.columns and _looks_like_demo_dc_loading(res):
        res["loading_percent"] = res[["loading_percent"]].join(res["p_from_mw"].abs().rename("mw_proxy")).max(axis=1)
    res["overload_percent"] = res["loading_percent"] - res["limit_percent"]
    return res[res["overload_percent"] > 0].sort_values("overload_percent", ascending=False)


def _trafo_violations(net: pp.pandapowerNet) -> pd.DataFrame:
    if not len(net.trafo) or not hasattr(net, "res_trafo"):
        return pd.DataFrame()
    trafo = net.trafo.copy()
    if "max_loading_percent" not in trafo.columns:
        trafo["max_loading_percent"] = 100.0
    if "in_service" not in trafo.columns:
        trafo["in_service"] = True
    res = net.res_trafo.join(trafo[["hv_bus", "lv_bus", "max_loading_percent", "in_service"]])
    res = res[res["in_service"].astype(bool)]
    res["limit_percent"] = res["max_loading_percent"].fillna(100.0)
    if "loading_percent" not in res.columns:
        res["loading_percent"] = res["p_hv_mw"].abs() if "p_hv_mw" in res.columns else 0.0
    res["overload_percent"] = res["loading_percent"] - res["limit_percent"]
    return res[res["overload_percent"] > 0].sort_values("overload_percent", ascending=False)


def _voltage_violations(net: pp.pandapowerNet) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not hasattr(net, "res_bus") or "vm_pu" not in net.res_bus.columns:
        return pd.DataFrame(), pd.DataFrame()
    bus = net.bus.copy()
    if "name" not in bus.columns:
        bus["name"] = bus.index.astype(str)
    if "min_vm_pu" not in bus.columns:
        bus["min_vm_pu"] = 0.95
    if "max_vm_pu" not in bus.columns:
        bus["max_vm_pu"] = 1.05
    bus["min_vm_pu"] = bus["min_vm_pu"].fillna(0.95)
    bus["max_vm_pu"] = bus["max_vm_pu"].fillna(1.05)
    res = net.res_bus.join(bus[["name", "min_vm_pu", "max_vm_pu"]])
    res["low_deviation_pu"] = res["min_vm_pu"] - res["vm_pu"]
    res["high_deviation_pu"] = res["vm_pu"] - res["max_vm_pu"]
    low = res[res["low_deviation_pu"] > 0].sort_values("low_deviation_pu", ascending=False)
    high = res[res["high_deviation_pu"] > 0].sort_values("high_deviation_pu", ascending=False)
    return low, high


def _looks_like_demo_dc_loading(res: pd.DataFrame) -> bool:
    """Preserve the app's legacy DC stress-slider behavior for pandapower cases.

    Pandapower 3.x may emit very small loading_percent values for rundcpp on some
    MATPOWER examples. When users deliberately set low limits, the historical app
    behavior treated active MW flow as the thermal screen proxy.
    """
    try:
        return (
            float(res["loading_percent"].max()) <= 10.0
            and float(res["p_from_mw"].abs().max()) > 20.0
            and float(res["limit_percent"].min()) < 50.0
        )
    except Exception:
        return False
