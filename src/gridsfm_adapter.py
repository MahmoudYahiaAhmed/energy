from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib.util
import logging
from pathlib import Path
from typing import Any

import pandapower as pp

from gridsfm_mapper import pandapower_net_to_gridsfm_pyg_json, validate_gridsfm_payload

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GridSFMPrediction:
    action: Any
    predicted_score: float
    feasibility: float
    predicted_violations: int
    failed: bool = False
    reason: str | None = None


class GridSFMAdapter:
    def __init__(
        self,
        checkpoint_path: str | None,
        device: str,
        min_buses: int = 500,
        feasibility_threshold: float = 0.80,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.min_buses = min_buses
        self.feasibility_threshold = feasibility_threshold
        self._model: Any | None = None
        self._load_error: str | None = None

    def is_available(self) -> bool:
        if importlib.util.find_spec("gridsfm") is None:
            self._load_error = "GridSFM package is not installed"
            return False
        if importlib.util.find_spec("torch") is None:
            self._load_error = "PyTorch is not installed"
            return False
        if self.checkpoint_path and not Path(self.checkpoint_path).exists():
            self._load_error = f"checkpoint not found: {self.checkpoint_path}"
            return False
        return True

    def is_supported_net(self, net: pp.pandapowerNet) -> tuple[bool, str]:
        n_bus = int(len(net.bus))
        if n_bus < self.min_buses:
            return (
                False,
                f"network has {n_bus} buses, below supported minimum {self.min_buses}",
            )
        if not self.is_available():
            return False, self._load_error or "GridSFM unavailable"
        try:
            payload = pandapower_net_to_gridsfm_pyg_json(net)
            validate_gridsfm_payload(payload)
        except Exception as exc:
            return False, f"pandapower-to-GridSFM mapping failed: {exc}"
        return True, "supported"

    def rank_candidates(
        self,
        base_net: pp.pandapowerNet,
        candidates: list[Any],
        config: Any | None = None,
    ) -> list[GridSFMPrediction]:
        predictions = self.predict_candidates(base_net, candidates, config)
        predictions.sort(key=lambda item: (item.failed, item.predicted_score, item.action.cost))
        return predictions

    def predict_candidates(
        self,
        base_net: pp.pandapowerNet,
        candidates: list[Any],
        config: Any | None = None,
    ) -> list[GridSFMPrediction]:
        predictions: list[GridSFMPrediction] = []
        for action in candidates:
            trial = deepcopy(base_net)
            try:
                action.apply(trial)
                payload = pandapower_net_to_gridsfm_pyg_json(trial)
                validate_gridsfm_payload(payload)
                prediction = self._predict_payload(action, payload)
            except Exception as exc:
                logger.debug("GridSFM candidate conversion failed: %s", exc)
                prediction = GridSFMPrediction(
                    action=action,
                    predicted_score=1_000_000_000.0 + float(getattr(action, "cost", 0.0)),
                    feasibility=0.0,
                    predicted_violations=999,
                    failed=True,
                    reason=str(exc),
                )
            predictions.append(prediction)
        return predictions

    def _predict_payload(self, action: Any, payload: dict[str, Any]) -> GridSFMPrediction:
        model = self._load_model()
        if model is None:
            # The optimizer still validates every candidate with pandapower. This fallback only
            # preserves deterministic ordering when GridSFM is installed incompletely.
            return GridSFMPrediction(
                action=action,
                predicted_score=float(getattr(action, "cost", 0.0)) + 100.0 * getattr(action, "disruptive_rank", 0),
                feasibility=0.0,
                predicted_violations=0,
                failed=True,
                reason=self._load_error or "GridSFM model unavailable",
            )

        # GridSFM package APIs have changed across releases. Keep this isolated and never use
        # the result as a safety authority; pandapower validation remains mandatory.
        raise RuntimeError("Installed GridSFM inference API is not wired for this environment")

    def _load_model(self) -> Any | None:
        if self._model is not None:
            return self._model
        if not self.is_available():
            return None
        try:
            from gridsfm import load_model  # type: ignore

            self._model = load_model(self.checkpoint_path, device=self.device)
            return self._model
        except Exception as exc:
            self._load_error = f"GridSFM model load failed: {exc}"
            logger.warning("%s", self._load_error)
            return None
