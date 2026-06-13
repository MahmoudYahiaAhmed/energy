from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd
import pandapower as pp


BUNDLED_SMARD_CSV = Path(__file__).resolve().parent.parent / "data" / "smard_de_load_sample_quarterhour.csv"


@dataclass(frozen=True)
class SmardSnapshot:
    timestamp: str
    load_mw: float | None
    generation_mw: float | None
    load_multiplier: float | None
    generation_multiplier: float | None


LOAD_KEYWORDS = ("netzlast", "stromverbrauch", "verbrauch", "load")
GENERATION_KEYWORDS = (
    "erzeugung",
    "generation",
    "wind",
    "solar",
    "photovoltaik",
    "biomasse",
    "wasserkraft",
    "nuclear",
    "kernenergie",
    "braunkohle",
    "steinkohle",
    "erdgas",
)


def read_smard_csv(file: BinaryIO | bytes) -> pd.DataFrame:
    """Read a SMARD CSV export and normalize it to timestamp/load/generation columns."""
    raw = file if isinstance(file, bytes) else file.read()
    source = BytesIO(raw)
    frame = pd.read_csv(source, sep=None, engine="python", encoding="utf-8-sig")
    if len(frame.columns) == 1:
        source.seek(0)
        frame = pd.read_csv(source, sep=";", encoding="utf-8-sig")

    numeric = {
        column: _to_number_series(frame[column])
        for column in frame.columns
        if _looks_numeric(frame[column])
    }
    timestamp = _timestamp_series(frame)
    load_column = _first_matching_column(numeric, LOAD_KEYWORDS)
    generation_columns = [
        column
        for column in numeric
        if _matches(column, GENERATION_KEYWORDS) and column != load_column
    ]

    normalized = pd.DataFrame({"timestamp": timestamp})
    normalized["load_mw"] = numeric[load_column] if load_column else None
    if generation_columns:
        normalized["generation_mw"] = pd.concat(
            [numeric[column] for column in generation_columns],
            axis=1,
        ).sum(axis=1, min_count=1)
    else:
        normalized["generation_mw"] = None

    if "load_mw" in normalized and normalized["load_mw"].notna().any():
        reference_load = float(normalized["load_mw"].dropna().median())
        normalized["load_multiplier"] = normalized["load_mw"] / reference_load if reference_load > 0 else None
    else:
        normalized["load_multiplier"] = None
    if "generation_mw" in normalized and normalized["generation_mw"].notna().any():
        reference_generation = float(normalized["generation_mw"].dropna().median())
        normalized["generation_multiplier"] = (
            normalized["generation_mw"] / reference_generation if reference_generation > 0 else None
        )
    else:
        normalized["generation_multiplier"] = None

    return normalized.dropna(how="all", subset=["load_mw", "generation_mw"]).reset_index(drop=True)


def read_bundled_smard_data() -> pd.DataFrame:
    return read_smard_csv(BUNDLED_SMARD_CSV.read_bytes())


def snapshot_from_row(row: pd.Series) -> SmardSnapshot:
    load = row.get("load_mw")
    generation = row.get("generation_mw")
    return SmardSnapshot(
        timestamp=str(row.get("timestamp", "")),
        load_mw=float(load) if pd.notna(load) else None,
        generation_mw=float(generation) if pd.notna(generation) else None,
        load_multiplier=float(row.get("load_multiplier")) if pd.notna(row.get("load_multiplier")) else None,
        generation_multiplier=(
            float(row.get("generation_multiplier")) if pd.notna(row.get("generation_multiplier")) else None
        ),
    )


def apply_smard_snapshot(net: pp.pandapowerNet, snapshot: SmardSnapshot) -> None:
    """Scale a pandapower case by the SMARD profile at one timestamp.

    SMARD values are German aggregate system values, while IEEE pandapower cases are much smaller
    synthetic networks. The compatible mapping is therefore a profile multiplier, not direct MW
    replacement.
    """
    if snapshot.load_multiplier is not None and snapshot.load_multiplier > 0 and len(net.load):
        net.load["p_mw"] = net.load["p_mw"] * snapshot.load_multiplier

    if snapshot.generation_multiplier is not None and snapshot.generation_multiplier > 0 and len(net.gen):
        net.gen["p_mw"] = net.gen["p_mw"] * snapshot.generation_multiplier


def _timestamp_series(frame: pd.DataFrame) -> pd.Series:
    date_column = _find_column(frame.columns, ("datum", "date"))
    time_column = _find_column(frame.columns, ("anfang", "beginn", "start", "time"))
    if date_column and time_column and date_column != time_column:
        return frame[date_column].astype(str) + " " + frame[time_column].astype(str)
    if date_column:
        return frame[date_column].astype(str)
    return pd.Series([f"row {idx}" for idx in frame.index], index=frame.index)


def _to_number_series(series: pd.Series) -> pd.Series:
    return series.map(_to_number)


def _to_number(value: object) -> float | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    return pd.to_numeric(text, errors="coerce")


def _looks_numeric(series: pd.Series) -> bool:
    return _to_number_series(series).notna().any()


def _first_matching_column(columns: dict[str, pd.Series], keywords: tuple[str, ...]) -> str | None:
    for column in columns:
        if _matches(column, keywords):
            return column
    return None


def _find_column(columns: pd.Index, keywords: tuple[str, ...]) -> str | None:
    for column in columns:
        if _matches(column, keywords):
            return str(column)
    return None


def _matches(column: object, keywords: tuple[str, ...]) -> bool:
    normalized = str(column).lower()
    return any(keyword in normalized for keyword in keywords)
