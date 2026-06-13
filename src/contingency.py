from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import pandapower as pp


ContingencyType = Literal["line", "generator"]


@dataclass(frozen=True)
class Contingency:
    kind: ContingencyType
    element_index: int
    label: str


def list_contingencies(net: pp.pandapowerNet) -> list[Contingency]:
    contingencies: list[Contingency] = []
    for idx, row in net.line.iterrows():
        if bool(row.get("in_service", True)):
            name = row.get("name", None)
            label = f"Line {idx}: bus {row.from_bus} -> bus {row.to_bus}"
            if name not in (None, ""):
                label = f"{label} ({name})"
            contingencies.append(Contingency("line", int(idx), label))

    for idx, row in net.gen.iterrows():
        if bool(row.get("in_service", True)):
            bus = int(row.bus)
            contingencies.append(Contingency("generator", int(idx), f"Generator {idx} at bus {bus}"))

    return contingencies


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
