from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GridSFMResult:
    case_name: str
    bus_count: int
    gen_count: int
    line_count: int
    has_solution: bool
    feasible: bool
    termination_status: str
    sample_path: str


# All cases that have a .pyg.json sample file
GRIDSFM_CASES: tuple[str, ...] = (
    "case1354_pegase", "case1803_snem", "case1888_rte", "case1951_rte",
    "case2000_goc", "case2312_goc", "case2383wp_k", "case2736sp_k",
    "case2737sop_k", "case2742_goc", "case2746wop_k", "case2746wp_k",
    "case2848_rte", "case2853_sdet", "case2868_rte", "case2869_pegase",
    "case3012wp_k", "case3022_goc", "case3120sp_k", "case3375wp_k",
    "case500_goc", "case588_sdet", "case793_goc",
    "msr_arizona", "msr_arkansas", "msr_california", "msr_colorado",
    "msr_desert_sw", "msr_florida", "msr_georgia", "msr_illinois",
    "msr_indiana", "msr_iowa", "msr_kansas", "msr_kentucky",
    "msr_louisiana", "msr_michigan", "msr_minnesota", "msr_mississippi",
    "msr_missouri", "msr_new_england", "msr_new_york", "msr_north_carolina",
    "msr_ohio", "msr_oklahoma", "msr_pacific_nw", "msr_pennsylvania",
    "msr_south_carolina", "msr_tennessee", "msr_texas", "msr_virginia",
    "msr_washington", "msr_wisconsin",
)

PANDAPOWER_AND_GRIDSFM: frozenset[str] = frozenset((
    "case1354_pegase", "case1888_rte", "case2848_rte", "case2869_pegase",
    "case2736sp_k", "case2737sop_k", "case2746wop_k", "case2746wp_k",
    "case3012wp_k", "case3120sp_k", "case3375wp_k",
))

_SOLVED_STATUSES = {"LOCALLY_SOLVED", "OPTIMAL", "ALMOST_OPTIMAL"}


class GridSFMService:
    """Reads pre-computed GridSFM sample files directly — no subprocess needed."""

    def __init__(self, samples_dir: str) -> None:
        self._samples_dir = samples_dir
        self._cache: dict[str, GridSFMResult] = {}

    @property
    def available(self) -> bool:
        return bool(self._samples_dir) and os.path.isdir(self._samples_dir)

    def _sample_path(self, case_name: str) -> Path:
        return Path(self._samples_dir) / f"{case_name}.pyg.json"

    def _load_case(self, case_name: str) -> GridSFMResult | None:
        path = self._sample_path(case_name)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                obj = json.load(f)
        except Exception:
            return None

        grid = obj.get("grid", {})
        nodes = grid.get("nodes", {})
        edges = grid.get("edges", {})
        bus_count = len(nodes.get("bus", []))
        gen_count = len(nodes.get("generator", []))
        line_count = len(edges.get("line", {}).get("edge_index", [[]])[0]) if "line" in edges else 0

        sol = obj.get("solution", {})
        has_solution = bool(sol)
        md = obj.get("metadata", {})
        ts = md.get("termination_status", "UNKNOWN")
        feasible = ts.upper() in _SOLVED_STATUSES if ts else bool(md.get("feasible", False))

        return GridSFMResult(
            case_name=case_name,
            bus_count=bus_count,
            gen_count=gen_count,
            line_count=line_count,
            has_solution=has_solution,
            feasible=feasible,
            termination_status=ts or "UNKNOWN",
            sample_path=str(path),
        )

    def get_case(self, case_name: str) -> GridSFMResult | None:
        if case_name not in self._cache:
            result = self._load_case(case_name)
            if result is None:
                return None
            self._cache[case_name] = result
        return self._cache[case_name]

    def list_available_cases(self) -> list[str]:
        if not self.available:
            return []
        return [
            p.stem  # e.g. "case1888_rte" from "case1888_rte.pyg.json"
            for p in sorted(Path(self._samples_dir).glob("*.pyg.json"))
        ]



