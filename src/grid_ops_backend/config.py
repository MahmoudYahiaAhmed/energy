from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
except Exception:
    pass


@dataclass(frozen=True)
class Settings:
    cors_allow_origins: tuple[str, ...]
    gridsfm_samples_dir: str


def load_settings() -> Settings:
    raw_origins = os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173,http://127.0.0.1:5174,http://127.0.0.1:5175",
    )
    origins = tuple(part.strip() for part in raw_origins.split(",") if part.strip())
    samples_dir = os.getenv("GRIDSFM_SAMPLES_DIR", "")
    if not samples_dir:
        model_dir = os.getenv("GRIDSFM_MODEL_DIR", "")
        if model_dir:
            samples_dir = os.path.join(model_dir, "samples")
    return Settings(
        cors_allow_origins=origins,
        gridsfm_samples_dir=samples_dir,
    )
