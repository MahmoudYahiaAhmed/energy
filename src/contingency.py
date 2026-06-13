from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import pandapower as pp
import pandas as pd


ContingencyType = Literal["line", "generator"]


@dataclass(frozen=True)
class Contingency:
    kind: ContingencyType
    element_index: int
    label: str


def list_contingencies(net: pp.pandapowerNet) -> list[Contingency]:
    line_rows = net.line.loc[_in_service_mask(net.line)]
    line_indices = line_rows.index.to_numpy(dtype=int)
    from_buses = line_rows["from_bus"].to_numpy(dtype=int)
    to_buses = line_rows["to_bus"].to_numpy(dtype=int)
    names = (
        line_rows["name"].fillna("").astype(str).to_numpy()
        if "name" in line_rows.columns
        else [""] * len(line_rows)
    )

    line_contingencies = [
        Contingency(
            "line",
            int(idx),
            f"Line {idx}: bus {from_bus} -> bus {to_bus}{f' ({name})' if name else ''}",
        )
        for idx, from_bus, to_bus, name in zip(line_indices, from_buses, to_buses, names)
    ]

    gen_rows = net.gen.loc[_in_service_mask(net.gen)]
    gen_indices = gen_rows.index.to_numpy(dtype=int)
    buses = gen_rows["bus"].to_numpy(dtype=int)
    gen_contingencies = [
        Contingency("generator", int(idx), f"Generator {idx} at bus {bus}")
        for idx, bus in zip(gen_indices, buses)
    ]

    return line_contingencies + gen_contingencies


def _in_service_mask(frame: pd.DataFrame) -> pd.Series:
    if "in_service" not in frame.columns:
        return pd.Series(True, index=frame.index)
    return frame["in_service"].fillna(True).astype(bool)


def apply_contingency(net: pp.pandapowerNet, contingency: Contingency) -> pp.pandapowerNet:
    outaged = deepcopy(net)
    if contingency.kind == "line":
        if contingency.element_index not in outaged.line.index:
            raise KeyError(f"Line {contingency.element_index} does not exist")
        outaged.line.at[contingency.element_index, "in_service"] = False
    elif contingency.kind == "generator":
        if contingency.element_index not in outaged.gen.index:
            raise KeyError(f"Generator {contingency.element_index} does not exist")
        outaged.gen.at[contingency.element_index, "in_service"] = False
    else:
        raise ValueError(f"Unsupported contingency kind: {contingency.kind}")
    return outaged
