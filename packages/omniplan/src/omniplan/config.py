"""Runtime settings, loaded from environment variables / a root-level .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[4]
load_dotenv(REPO_ROOT / ".env")


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("OMNIPLAN_DATA_DIR", REPO_ROOT / "data")))
    horizon_weeks: int = field(default_factory=lambda: _int("OMNIPLAN_HORIZON_WEEKS", 12))
    max_cover_days: int = field(default_factory=lambda: _int("OMNIPLAN_MAX_COVER_DAYS", 120))
    default_lead_time_weeks: int = field(default_factory=lambda: _int("OMNIPLAN_DEFAULT_LEAD_TIME_WEEKS", 4))
    llm_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    llm_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", ""))
    llm_temperature: float = field(default_factory=lambda: _float("OPENAI_TEMPERATURE", 0.3))

    @property
    def pos_path(self) -> Path:
        return self.data_dir / "pos_weekly_data.csv"

    @property
    def master_path(self) -> Path:
        return self.data_dir / "master_sku_data.csv"

    @property
    def promo_calendar_path(self) -> Path:
        return self.data_dir / "promo_calendar.csv"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)


def get_settings() -> Settings:
    return Settings()
