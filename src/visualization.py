from __future__ import annotations

import math

import networkx as nx
import pandas as pd
import pandapower as pp
import plotly.graph_objects as go


def network_figure(net: pp.pandapowerNet, title: str = "Grid topology") -> go.Figure:
    graph = nx.Graph()
    for bus_idx in net.bus.index:
        graph.add_node(int(bus_idx))
    for line_idx, row in net.line.iterrows():
        graph.add_edge(int(row.from_bus), int(row.to_bus))
    for trafo_idx, row in net.trafo.iterrows():
        graph.add_edge(int(row.hv_bus), int(row.lv_bus))

    pos = nx.spring_layout(graph, seed=8) if len(graph) else {}
    fig = go.Figure()
    branch_counts = _branch_counts(net)
    branch_seen: dict[tuple[int, int], int] = {}

    for line_idx, row in net.line.iterrows():
        u = int(row.from_bus)
        v = int(row.to_bus)
        order = _next_branch_order(branch_seen, u, v)
        _add_branch_trace(
            fig=fig,
            net=net,
            pos=pos,
            u=u,
            v=v,
            element="line",
            index=int(line_idx),
            in_service=bool(row.get("in_service", True)),
            order=order,
            total=branch_counts[_branch_key(u, v)],
        )

    for trafo_idx, row in net.trafo.iterrows():
        u = int(row.hv_bus)
        v = int(row.lv_bus)
        order = _next_branch_order(branch_seen, u, v)
        _add_branch_trace(
            fig=fig,
            net=net,
            pos=pos,
            u=u,
            v=v,
            element="trafo",
            index=int(trafo_idx),
            in_service=bool(row.get("in_service", True)),
            order=order,
            total=branch_counts[_branch_key(u, v)],
        )

    bus_x = [pos[node][0] for node in graph.nodes]
    bus_y = [pos[node][1] for node in graph.nodes]
    bus_ids = list(graph.nodes)
    bus_colors = [_bus_color(net, bus) for bus in bus_ids]
    bus_text = [_bus_hover(net, bus) for bus in bus_ids]

    fig.add_trace(
        go.Scatter(
            x=bus_x,
            y=bus_y,
            mode="markers+text",
            marker=dict(size=16, color=bus_colors, line=dict(width=1, color="#222")),
            text=[str(bus) for bus in bus_ids],
            textposition="top center",
            hovertext=bus_text,
            hoverinfo="text",
            showlegend=False,
        )
    )
    _add_generator_traces(fig, net, pos)

    fig.update_layout(
        title=title,
        height=480,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
    )
    return fig


def _add_branch_trace(
    fig: go.Figure,
    net: pp.pandapowerNet,
    pos: dict[int, tuple[float, float]],
    u: int,
    v: int,
    element: str,
    index: int,
    in_service: bool,
    order: int,
    total: int,
) -> None:
    if u not in pos or v not in pos:
        return

    x_values, y_values = _branch_coordinates(pos[u], pos[v], order, total)
    if in_service:
        color = _line_color(net, {"element": element, "index": index})
        dash = "solid"
        width = 4 if color == "red" else 2.5 if color == "orange" else 1.5
        status = "in service"
    else:
        color = "#c62828"
        dash = "dash"
        width = 3
        status = "out of service"

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hoverinfo="text",
            text=f"{element} {index}: {u} - {v}<br>{status}",
            showlegend=False,
        )
    )


def _add_generator_traces(fig: go.Figure, net: pp.pandapowerNet, pos: dict[int, tuple[float, float]]) -> None:
    if not len(net.gen):
        return

    for gen_idx, row in net.gen.iterrows():
        bus = int(row.bus)
        if bus not in pos:
            continue
        x, y = pos[bus]
        in_service = bool(row.get("in_service", True))
        fig.add_trace(
            go.Scatter(
                x=[x + 0.035],
                y=[y - 0.035],
                mode="markers",
                marker=dict(
                    size=13,
                    symbol="triangle-up" if in_service else "x",
                    color="#1565c0" if in_service else "#c62828",
                    line=dict(width=1, color="#222"),
                ),
                hoverinfo="text",
                text=f"Generator {int(gen_idx)} at bus {bus}<br>{'in service' if in_service else 'out of service'}",
                showlegend=False,
            )
        )


def _branch_counts(net: pp.pandapowerNet) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for _, row in net.line.iterrows():
        key = _branch_key(int(row.from_bus), int(row.to_bus))
        counts[key] = counts.get(key, 0) + 1
    for _, row in net.trafo.iterrows():
        key = _branch_key(int(row.hv_bus), int(row.lv_bus))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _next_branch_order(seen: dict[tuple[int, int], int], u: int, v: int) -> int:
    key = _branch_key(u, v)
    order = seen.get(key, 0)
    seen[key] = order + 1
    return order


def _branch_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def _branch_coordinates(
    start: tuple[float, float],
    end: tuple[float, float],
    order: int,
    total: int,
) -> tuple[list[float], list[float]]:
    x1, y1 = start
    x2, y2 = end
    if total <= 1:
        return [x1, x2], [y1, y2]

    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy) or 1.0
    offset = (order - (total - 1) / 2.0) * 0.035
    ox = -dy / length * offset
    oy = dx / length * offset
    return [x1 + ox, x2 + ox], [y1 + oy, y2 + oy]


def bus_voltage_table(net: pp.pandapowerNet) -> pd.DataFrame:
    if not hasattr(net, "res_bus"):
        return pd.DataFrame()
    columns = [column for column in ["va_degree", "p_mw"] if column in net.res_bus.columns]
    table = net.res_bus[columns].join(net.bus[["name"]])
    table["status"] = "dc solved"
    return table.reset_index(names="bus")


def line_loading_table(net: pp.pandapowerNet) -> pd.DataFrame:
    if not hasattr(net, "res_line"):
        return pd.DataFrame()
    result_columns = [column for column in ["loading_percent", "p_from_mw", "p_to_mw"] if column in net.res_line.columns]
    table = net.res_line[result_columns].join(
        net.line[["from_bus", "to_bus", "max_loading_percent", "in_service"]]
    )
    if "loading_percent" not in table.columns:
        table["loading_percent"] = table["p_from_mw"].abs() if "p_from_mw" in table.columns else 0.0
    table["status"] = table.apply(_line_status_row, axis=1)
    return table.reset_index(names="line")


def generator_table(net: pp.pandapowerNet) -> pd.DataFrame:
    frames = []
    if len(net.ext_grid) and hasattr(net, "res_ext_grid"):
        ext_columns = [column for column in ["p_mw", "q_mvar"] if column in net.res_ext_grid.columns]
        ext = net.ext_grid[["bus", "in_service"]].join(net.res_ext_grid[ext_columns])
        ext["type"] = "ext_grid"
        ext["element"] = ext.index
        frames.append(ext)
    if len(net.gen) and hasattr(net, "res_gen"):
        gen_columns = [column for column in ["p_mw", "q_mvar", "vm_pu"] if column in net.res_gen.columns]
        gen = net.gen[["bus", "p_mw", "in_service"]].join(net.res_gen[gen_columns], rsuffix="_result")
        gen["type"] = "gen"
        gen["element"] = gen.index
        frames.append(gen)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _line_color(net: pp.pandapowerNet, data: dict) -> str:
    if data["element"] == "trafo":
        if not hasattr(net, "res_trafo") or data["index"] not in net.res_trafo.index:
            return "gray"
        loading = float(net.res_trafo.at[data["index"], "loading_percent"])
        limit = float(net.trafo.at[data["index"], "max_loading_percent"])
    else:
        if not hasattr(net, "res_line") or data["index"] not in net.res_line.index:
            return "gray"
        loading = float(net.res_line.at[data["index"], "loading_percent"])
        limit = float(net.line.at[data["index"], "max_loading_percent"])

    if math.isnan(loading):
        return "gray"
    if loading > limit:
        return "red"
    if loading > 0.85 * limit:
        return "orange"
    return "gray"


def _bus_color(net: pp.pandapowerNet, bus: int) -> str:
    if not hasattr(net, "res_bus") or bus not in net.res_bus.index:
        return "gray"
    return "green"


def _bus_hover(net: pp.pandapowerNet, bus: int) -> str:
    name = net.bus.at[bus, "name"] if "name" in net.bus.columns else bus
    if hasattr(net, "res_bus") and bus in net.res_bus.index:
        if "va_degree" in net.res_bus.columns:
            return f"Bus {bus} ({name})<br>Angle: {net.res_bus.at[bus, 'va_degree']:.3f} deg"
    return f"Bus {bus} ({name})"


def _line_status_row(row: pd.Series) -> str:
    if not bool(row.in_service):
        return "out"
    if row.loading_percent > row.max_loading_percent:
        return "overloaded"
    if row.loading_percent > 0.85 * row.max_loading_percent:
        return "near limit"
    return "normal"
