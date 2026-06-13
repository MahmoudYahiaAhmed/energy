from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandapower as pp


BUS_WIDTH = 4
GENERATOR_WIDTH = 11
LOAD_WIDTH = 2
SHUNT_WIDTH = 2
AC_LINE_WIDTH = 7
TRANSFORMER_WIDTH = 11


def pandapower_net_to_gridsfm_pyg_json(
    net: pp.pandapowerNet,
    path: str | Path | None = None,
) -> dict[str, Any]:
    base_mva = float(getattr(net, "sn_mva", 1.0) or 1.0)
    bus_rows = _in_service_bus_rows(net)
    bus_id_map = {int(bus_idx): pos for pos, bus_idx in enumerate(bus_rows.index.tolist())}
    slack_buses = set(int(row.bus) for _, row in _in_service_rows(net.ext_grid).iterrows()) if len(net.ext_grid) else set()
    pv_buses = set(int(row.bus) for _, row in _in_service_rows(net.gen).iterrows()) if len(net.gen) else set()

    buses = []
    for bus_idx, row in bus_rows.iterrows():
        bus_idx = int(bus_idx)
        bus_type = 3 if bus_idx in slack_buses else 2 if bus_idx in pv_buses else 1
        buses.append([
            _finite_float(row.get("vn_kv", 0.0)),
            float(bus_type),
            _voltage_limit(row.get("min_vm_pu"), 0.95),
            _voltage_limit(row.get("max_vm_pu"), 1.05),
        ])

    generators, generator_links, gen_reverse = _generator_nodes(net, bus_id_map, base_mva)
    loads, load_links, load_reverse = _load_nodes(net, bus_id_map, base_mva)
    shunts, shunt_links, shunt_reverse = _shunt_nodes(net, bus_id_map, base_mva)
    ac_lines, ac_line_index, line_reverse = _line_edges(net, bus_id_map)
    trafos, trafo_index, trafo_reverse = _trafo_edges(net, bus_id_map)

    payload: dict[str, Any] = {
        "grid": {
            "baseMVA": base_mva,
            "nodes": {
                "bus": buses,
                "generator": generators,
                "load": loads,
                "shunt": shunts,
            },
            "edges": {
                "ac_line": {"edge_index": ac_line_index, "edge_attr": ac_lines},
                "transformer": {"edge_index": trafo_index, "edge_attr": trafos},
                "generator_link": {"edge_index": generator_links, "edge_attr": [[0.0] for _ in generator_links[0]]},
                "load_link": {"edge_index": load_links, "edge_attr": [[0.0] for _ in load_links[0]]},
                "shunt_link": {"edge_index": shunt_links, "edge_attr": [[0.0] for _ in shunt_links[0]]},
            },
        },
        "metadata": {
            "schema_note": "Conservative GridSFM .pyg.json-like export; validate against installed GridSFM schema before inference.",
            "pandapower_mapping": {
                "bus": {str(v): int(k) for k, v in bus_id_map.items()},
                "generator": gen_reverse,
                "load": load_reverse,
                "shunt": shunt_reverse,
                "ac_line": line_reverse,
                "transformer": trafo_reverse,
            },
        },
    }
    validate_gridsfm_payload(payload)
    if path is not None:
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def validate_gridsfm_payload(payload: dict[str, Any]) -> None:
    grid = payload.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("payload must contain grid object")
    nodes = grid.get("nodes")
    edges = grid.get("edges")
    if not isinstance(nodes, dict) or not isinstance(edges, dict):
        raise ValueError("payload grid must contain nodes and edges")

    required_nodes = {
        "bus": BUS_WIDTH,
        "generator": GENERATOR_WIDTH,
        "load": LOAD_WIDTH,
        "shunt": SHUNT_WIDTH,
    }
    for name, width in required_nodes.items():
        rows = nodes.get(name)
        if rows is None:
            raise ValueError(f"missing node block {name}")
        if name == "bus" and not rows:
            raise ValueError("bus node block must be non-empty")
        _validate_rows(f"nodes.{name}", rows, width)

    slack_count = sum(1 for row in nodes["bus"] if int(row[1]) == 3)
    if slack_count < 1:
        raise ValueError("at least one slack/reference bus is required")

    required_edges = {
        "ac_line": AC_LINE_WIDTH,
        "transformer": TRANSFORMER_WIDTH,
        "generator_link": 1,
        "load_link": 1,
        "shunt_link": 1,
    }
    node_sizes = {name: len(rows) for name, rows in nodes.items()}
    for name, attr_width in required_edges.items():
        edge = edges.get(name)
        if not isinstance(edge, dict):
            raise ValueError(f"missing edge block {name}")
        edge_index = edge.get("edge_index")
        edge_attr = edge.get("edge_attr", [])
        if not _valid_edge_index(edge_index):
            raise ValueError(f"{name}.edge_index must contain [senders, receivers]")
        if len(edge_index[0]) != len(edge_index[1]):
            raise ValueError(f"{name} senders/receivers length mismatch")
        if len(edge_attr) != len(edge_index[0]):
            raise ValueError(f"{name} edge_attr length mismatch")
        _validate_rows(f"edges.{name}.edge_attr", edge_attr, attr_width)
        _validate_edge_ranges(name, edge_index, node_sizes)

    metadata = payload.get("metadata", {})
    mapping = metadata.get("pandapower_mapping") if isinstance(metadata, dict) else None
    if not isinstance(mapping, dict):
        raise ValueError("metadata.pandapower_mapping is required")
    for key in ("bus", "generator", "load", "shunt", "ac_line", "transformer"):
        if key not in mapping:
            raise ValueError(f"metadata.pandapower_mapping.{key} is required")


def _generator_nodes(
    net: pp.pandapowerNet,
    bus_id_map: dict[int, int],
    base_mva: float,
) -> tuple[list[list[float]], list[list[int]], dict[str, Any]]:
    rows: list[list[float]] = []
    edge_index = [[], []]
    reverse: dict[str, Any] = {}
    gen_id = 0

    for idx, row in _in_service_rows(net.gen).iterrows() if len(net.gen) else []:
        bus = int(row.bus)
        if bus not in bus_id_map:
            continue
        rows.append([
            base_mva,
            0.0,
            _mw(row.get("min_p_mw", 0.0), base_mva),
            _mw(row.get("max_p_mw", max(float(row.p_mw), 0.0) * 1.4 + 1.0), base_mva),
            0.0,
            _mw(row.get("min_q_mvar", -base_mva), base_mva),
            _mw(row.get("max_q_mvar", base_mva), base_mva),
            _finite_float(row.get("vm_pu", 1.0)),
            0.0,
            1.0,
            0.0,
        ])
        edge_index[0].append(gen_id)
        edge_index[1].append(bus_id_map[bus])
        reverse[str(gen_id)] = {"source": "gen", "pandapower_index": int(idx)}
        gen_id += 1

    for idx, row in _in_service_rows(net.ext_grid).iterrows() if len(net.ext_grid) else []:
        bus = int(row.bus)
        if bus not in bus_id_map:
            continue
        rows.append([
            base_mva,
            0.0,
            _mw(row.get("min_p_mw", -10.0 * base_mva), base_mva),
            _mw(row.get("max_p_mw", 10.0 * base_mva), base_mva),
            0.0,
            _mw(row.get("min_q_mvar", -10.0 * base_mva), base_mva),
            _mw(row.get("max_q_mvar", 10.0 * base_mva), base_mva),
            _finite_float(row.get("vm_pu", 1.0)),
            0.0,
            1.0,
            0.0,
        ])
        edge_index[0].append(gen_id)
        edge_index[1].append(bus_id_map[bus])
        reverse[str(gen_id)] = {"source": "ext_grid", "pandapower_index": int(idx)}
        gen_id += 1

    return rows, edge_index, reverse


def _load_nodes(
    net: pp.pandapowerNet,
    bus_id_map: dict[int, int],
    base_mva: float,
) -> tuple[list[list[float]], list[list[int]], dict[str, int]]:
    rows: list[list[float]] = []
    edge_index = [[], []]
    reverse: dict[str, int] = {}
    load_id = 0
    for idx, row in _in_service_rows(net.load).iterrows() if len(net.load) else []:
        bus = int(row.bus)
        if bus not in bus_id_map:
            continue
        rows.append([_mw(row.get("p_mw", 0.0), base_mva), _mw(row.get("q_mvar", 0.0), base_mva)])
        edge_index[0].append(load_id)
        edge_index[1].append(bus_id_map[bus])
        reverse[str(load_id)] = int(idx)
        load_id += 1
    return rows, edge_index, reverse


def _shunt_nodes(
    net: pp.pandapowerNet,
    bus_id_map: dict[int, int],
    base_mva: float,
) -> tuple[list[list[float]], list[list[int]], dict[str, int]]:
    if not hasattr(net, "shunt") or not len(net.shunt):
        return [], [[], []], {}
    rows: list[list[float]] = []
    edge_index = [[], []]
    reverse: dict[str, int] = {}
    shunt_id = 0
    for idx, row in _in_service_rows(net.shunt).iterrows():
        bus = int(row.bus)
        if bus not in bus_id_map:
            continue
        rows.append([_mw(row.get("p_mw", 0.0), base_mva), _mw(row.get("q_mvar", 0.0), base_mva)])
        edge_index[0].append(shunt_id)
        edge_index[1].append(bus_id_map[bus])
        reverse[str(shunt_id)] = int(idx)
        shunt_id += 1
    return rows, edge_index, reverse


def _line_edges(
    net: pp.pandapowerNet,
    bus_id_map: dict[int, int],
) -> tuple[list[list[float]], list[list[int]], dict[str, int]]:
    attrs: list[list[float]] = []
    edge_index = [[], []]
    reverse: dict[str, int] = {}
    edge_id = 0
    for idx, row in _in_service_rows(net.line).iterrows() if len(net.line) else []:
        from_bus = int(row.from_bus)
        to_bus = int(row.to_bus)
        if from_bus not in bus_id_map or to_bus not in bus_id_map:
            continue
        edge_index[0].append(bus_id_map[from_bus])
        edge_index[1].append(bus_id_map[to_bus])
        attrs.append(_line_attr(row, net, from_bus))
        reverse[str(edge_id)] = int(idx)
        edge_id += 1
    return attrs, edge_index, reverse


def _trafo_edges(
    net: pp.pandapowerNet,
    bus_id_map: dict[int, int],
) -> tuple[list[list[float]], list[list[int]], dict[str, int]]:
    attrs: list[list[float]] = []
    edge_index = [[], []]
    reverse: dict[str, int] = {}
    edge_id = 0
    for idx, row in _in_service_rows(net.trafo).iterrows() if len(net.trafo) else []:
        hv = int(row.hv_bus)
        lv = int(row.lv_bus)
        if hv not in bus_id_map or lv not in bus_id_map:
            continue
        edge_index[0].append(bus_id_map[hv])
        edge_index[1].append(bus_id_map[lv])
        attrs.append([
            -30.0,
            30.0,
            _finite_float(row.get("vkr_percent", 0.0)) / 100.0,
            _finite_float(row.get("vk_percent", 0.0)) / 100.0,
            _finite_float(row.get("sn_mva", 0.0)),
            0.0,
            0.0,
            _finite_float(row.get("tap_pos", 0.0)),
            _finite_float(row.get("shift_degree", 0.0)),
            0.0,
            0.0,
        ])
        reverse[str(edge_id)] = int(idx)
        edge_id += 1
    return attrs, edge_index, reverse


def _line_attr(row: Any, net: pp.pandapowerNet, from_bus: int) -> list[float]:
    base_kv = _finite_float(net.bus.at[from_bus, "vn_kv"] if "vn_kv" in net.bus.columns else 1.0)
    base_mva = float(getattr(net, "sn_mva", 1.0) or 1.0)
    z_base = base_kv * base_kv / base_mva if base_mva else 1.0
    length = _finite_float(row.get("length_km", 1.0), default=1.0)
    r = _finite_float(row.get("r_ohm_per_km", 0.0)) * length / z_base
    x = _finite_float(row.get("x_ohm_per_km", 0.0)) * length / z_base
    c_nf = _finite_float(row.get("c_nf_per_km", 0.0)) * length
    b = 2.0 * math.pi * 50.0 * c_nf * 1e-9 * z_base
    rate = _finite_float(row.get("max_i_ka", 0.0)) * base_kv * math.sqrt(3.0)
    return [-30.0, 30.0, b / 2.0, b / 2.0, r, x, rate]


def _validate_rows(name: str, rows: list[Any], width: int) -> None:
    for pos, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(f"{name}[{pos}] must have width {width}")
        for value in row:
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name}[{pos}] contains non-finite value")


def _valid_edge_index(edge_index: Any) -> bool:
    return (
        isinstance(edge_index, list)
        and len(edge_index) == 2
        and isinstance(edge_index[0], list)
        and isinstance(edge_index[1], list)
    )


def _validate_edge_ranges(name: str, edge_index: list[list[int]], node_sizes: dict[str, int]) -> None:
    if name == "generator_link":
        src_size, dst_size = node_sizes["generator"], node_sizes["bus"]
    elif name == "load_link":
        src_size, dst_size = node_sizes["load"], node_sizes["bus"]
    elif name == "shunt_link":
        src_size, dst_size = node_sizes["shunt"], node_sizes["bus"]
    else:
        src_size = dst_size = node_sizes["bus"]

    for sender in edge_index[0]:
        if not isinstance(sender, int) or sender < 0 or sender >= src_size:
            raise ValueError(f"{name} sender index out of range")
    for receiver in edge_index[1]:
        if not isinstance(receiver, int) or receiver < 0 or receiver >= dst_size:
            raise ValueError(f"{name} receiver index out of range")


def _in_service_bus_rows(net: pp.pandapowerNet):
    if "in_service" not in net.bus.columns:
        return net.bus
    return net.bus[net.bus["in_service"].fillna(True).astype(bool)]


def _in_service_rows(frame: Any):
    if "in_service" not in frame.columns:
        return frame
    return frame[frame["in_service"].fillna(True).astype(bool)]


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _voltage_limit(value: Any, default: float) -> float:
    return _finite_float(value, default=default)


def _mw(value: Any, base_mva: float) -> float:
    return _finite_float(value, 0.0) / base_mva
