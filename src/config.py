from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    base_url: str
    request_delay: float
    connect_timeout: float
    read_timeout: float
    max_retries: int
    data_dir: Path
    log_dir: Path
    api_key: str | None


def load_settings(path: Path = Path("config/settings.yaml")) -> Settings:
    load_dotenv()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    api, paths = raw["api"], raw["paths"]
    return Settings(
        base_url=str(api["base_url"]).rstrip("/"),
        request_delay=float(api["request_delay"]),
        connect_timeout=float(api["connect_timeout"]),
        read_timeout=float(api["read_timeout"]),
        max_retries=int(api["max_retries"]),
        data_dir=Path(paths["data_dir"]),
        log_dir=Path(paths["log_dir"]),
        api_key=os.getenv("OPENDOTA_API_KEY") or None,
    )

