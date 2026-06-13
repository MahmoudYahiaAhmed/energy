from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ScreeningMode = Literal["auto", "gridsfm", "dc", "pandapower"]


@dataclass(frozen=True)
class PipelineConfig:
    smard_csv_path: Path
    network_name: str
    top_k: int
    min_voltage_pu: float
    max_voltage_pu: float
    max_line_loading_percent: float
    screening: ScreeningMode
    use_gridsfm: bool
    cpu_only: bool
    use_lightsim2grid: bool
    output_folder: Path
    gridsfm_checkpoint: Path | None


def load_config(argv: list[str] | None = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Hybrid GridSFM/DC screening + pandapower validation pipeline")
    parser.add_argument("--smard", default="smard_load.csv", help="Path to SMARD CSV with timestamp/load_mw columns")
    parser.add_argument("--network", default="case118", help="case14, case30, case57, case118, or path to pandapower JSON")
    parser.add_argument("--top-k", type=int, default=50, help="Number of risky cases to validate with AC pandapower")
    parser.add_argument("--screening", choices=["auto", "gridsfm", "dc", "pandapower"], default="auto")
    parser.add_argument("--min-voltage", type=float, default=0.95)
    parser.add_argument("--max-voltage", type=float, default=1.05)
    parser.add_argument("--max-loading", type=float, default=100.0)
    parser.add_argument("--enable-gridsfm", action="store_true", help="Allow GridSFM screening when configured")
    parser.add_argument("--gridsfm-checkpoint", default="", help="Path to gridsfm_open_v1.1.pt checkpoint")
    parser.add_argument("--use-lightsim2grid", action="store_true", help="Use LightSim2Grid acceleration if available")
    parser.add_argument("--outputs", default="outputs", help="Folder for CSV reports and plots")
    args = parser.parse_args(argv)

    return PipelineConfig(
        smard_csv_path=Path(args.smard),
        network_name=args.network,
        top_k=max(1, int(args.top_k)),
        min_voltage_pu=args.min_voltage,
        max_voltage_pu=args.max_voltage,
        max_line_loading_percent=args.max_loading,
        screening=args.screening,
        use_gridsfm=bool(args.enable_gridsfm and args.screening in {"auto", "gridsfm"}),
        cpu_only=not bool(args.enable_gridsfm),
        use_lightsim2grid=bool(args.use_lightsim2grid),
        output_folder=Path(args.outputs),
        gridsfm_checkpoint=Path(args.gridsfm_checkpoint) if args.gridsfm_checkpoint else None,
    )
