from __future__ import annotations

import pandas as pd


OPTIONAL_COLUMNS = ("solar_mw", "wind_mw")


def load_smard_data(path: str) -> pd.DataFrame:
    """Load and clean SMARD.de time-series data.

    Required columns:
      - timestamp
      - load_mw

    Optional columns:
      - solar_mw
      - wind_mw
    """
    df = pd.read_csv(path)
    required = {"timestamp", "load_mw"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"SMARD CSV missing required column(s): {', '.join(sorted(missing))}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    numeric_columns = ["load_mw", *[column for column in OPTIONAL_COLUMNS if column in df.columns]]
    for column in numeric_columns:
        df[column] = _to_numeric_series(df[column])

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df[numeric_columns] = df[numeric_columns].interpolate(limit_direction="both")
    df = df.dropna(subset=["load_mw"])
    df = df[df["load_mw"] > 0].reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid SMARD rows remain after cleaning.")
    return df


def _to_numeric_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    has_comma_decimal = text.str.contains(",", regex=False)
    text = text.where(~has_comma_decimal, text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(text, errors="coerce")
