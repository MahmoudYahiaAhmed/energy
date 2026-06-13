from __future__ import annotations

import pandapower as pp


def generate_n1_candidates(net: pp.pandapowerNet) -> list[dict]:
    contingencies: list[dict] = []
    for idx, row in net.line.iterrows():
        if bool(row.get("in_service", True)):
            contingencies.append(
                {
                    "contingency_id": f"line-{int(idx)}",
                    "type": "line",
                    "element_id": int(idx),
                    "description": f"Open line {idx}: bus {row.from_bus} -> bus {row.to_bus}",
                }
            )
    for idx, row in net.trafo.iterrows():
        if bool(row.get("in_service", True)):
            contingencies.append(
                {
                    "contingency_id": f"trafo-{int(idx)}",
                    "type": "trafo",
                    "element_id": int(idx),
                    "description": f"Open transformer {idx}: bus {row.hv_bus} -> bus {row.lv_bus}",
                }
            )
    for idx, row in net.gen.iterrows():
        if bool(row.get("in_service", True)):
            contingencies.append(
                {
                    "contingency_id": f"gen-{int(idx)}",
                    "type": "gen",
                    "element_id": int(idx),
                    "description": f"Disconnect generator {idx} at bus {row.bus}",
                }
            )
    return contingencies
