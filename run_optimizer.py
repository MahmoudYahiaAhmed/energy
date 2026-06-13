from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contingency import list_contingencies
from corrective_agent import CorrectiveOptimizerConfig, optimize_post_contingency
from grid_loader import load_network


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hybrid post-contingency corrective optimization.")
    parser.add_argument("--network", default="case118", help="Pandapower network name or alias.")
    parser.add_argument("--contingency-index", type=int, default=0, help="Index into generated N-1 contingencies.")
    parser.add_argument("--use-gridsfm", action="store_true", help="Use GridSFM only to rank/screen candidates.")
    parser.add_argument("--gridsfm-checkpoint", default=None)
    parser.add_argument("--gridsfm-device", default="cpu")
    parser.add_argument("--gridsfm-screen-top-k", type=int, default=20)
    parser.add_argument("--max-greedy-steps", type=int, default=10)
    parser.add_argument("--allow-line-switching", action="store_true")
    parser.add_argument("--no-load-curtailment", action="store_true")
    parser.add_argument("--mode", choices=("ac", "dc"), default="ac")
    parser.add_argument("--output-report", default=None)
    parser.add_argument("--gridsfm-scenario", default=None, help="GridSFM .pyg.json inference-only scenario path.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    if args.gridsfm_scenario:
        print(
            "GridSFM scenario loading is inference-only in this CLI. "
            "No pandapower reconstruction/final solver validation is implemented for native .pyg.json scenarios."
        )
        return 2

    net = load_network(args.network)
    contingencies = list_contingencies(net)
    if not contingencies:
        raise SystemExit("No line or generator contingencies available for the selected network.")
    if args.contingency_index < 0 or args.contingency_index >= len(contingencies):
        raise SystemExit(f"--contingency-index must be between 0 and {len(contingencies) - 1}")

    config = CorrectiveOptimizerConfig(
        mode=args.mode,
        use_gridsfm=args.use_gridsfm,
        gridsfm_checkpoint=args.gridsfm_checkpoint,
        gridsfm_device=args.gridsfm_device,
        gridsfm_screen_top_k=args.gridsfm_screen_top_k,
        max_greedy_steps=args.max_greedy_steps,
        include_voltage=args.mode == "ac",
        allow_line_switching=args.allow_line_switching,
        allow_load_curtailment=not args.no_load_curtailment,
    )
    result = optimize_post_contingency(net, contingencies[args.contingency_index], config)
    report = {
        "status": result.status,
        "network": args.network,
        "contingency": contingencies[args.contingency_index].label,
        "initial_score": result.initial_score,
        "final_score": result.final_score,
        "initial_violations": result.initial_violations.to_summary(),
        "final_violations": result.final_violations.to_summary(),
        "gridsfm_used": result.gridsfm_used,
        "gridsfm_skip_reason": result.gridsfm_skip_reason,
        "actions": [
            {
                "type": action.action_type,
                "target_table": action.target_table,
                "target_index": action.target_index,
                "params": action.params,
                "cost": action.cost,
                "disruptive_rank": action.disruptive_rank,
                "reason": action.reason,
            }
            for action in result.actions
        ],
        "step_logs": result.step_logs,
    }
    text = json.dumps(report, indent=2)
    if args.output_report:
        Path(args.output_report).write_text(text, encoding="utf-8")
    print(text)
    return 0 if result.status in {"already_safe", "safe"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
