from __future__ import annotations

import logging
import time
from email.utils import parsedate_to_datetime
from typing import Any

import requests

from .config import Settings

RETRYABLE = {429, 500, 502, 503, 504}


class APIError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, attempts: int = 1):
        super().__init__(message)
        self.status, self.attempts = status, attempts


class OpenDotaClient:
    def __init__(self, settings: Settings, request_delay: float | None = None, max_retries: int | None = None):
        self.settings = settings
        self.delay = settings.request_delay if request_delay is None else request_delay
        self.max_retries = settings.max_retries if max_retries is None else max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "dota2-draft-analytics/1.0", "Accept": "application/json"})
        if settings.api_key:
            self.session.headers["Authorization"] = f"Bearer {settings.api_key}"
        self.last_request = 0.0
        self.request_count = 0

    def _wait(self) -> None:
        remaining = self.delay - (time.monotonic() - self.last_request)
        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def _retry_after(value: str | None) -> float | None:
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                return max(0.0, (parsedate_to_datetime(value) - parsedate_to_datetime(time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime()))).total_seconds())
            except (TypeError, ValueError):
                return None

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.settings.base_url}/{path.lstrip('/')}"
        for attempt in range(1, self.max_retries + 2):
            self._wait()
            try:
                response = self.session.get(url, params=params, timeout=(self.settings.connect_timeout, self.settings.read_timeout))
                self.last_request = time.monotonic()
                self.request_count += 1
            except requests.RequestException as exc:
                if attempt > self.max_retries:
                    raise APIError(str(exc), attempts=attempt) from exc
                time.sleep(min(60.0, 2 ** (attempt - 1)))
                continue
            if response.status_code in RETRYABLE and attempt <= self.max_retries:
                wait = self._retry_after(response.headers.get("Retry-After")) or min(60.0, 2 ** (attempt - 1))
                logging.warning("HTTP %s; retrying in %.1fs", response.status_code, wait)
                time.sleep(wait)
                continue
            if not response.ok:
                raise APIError(f"HTTP {response.status_code}: {response.text[:300]}", response.status_code, attempt)
            try:
                return response.json()
            except ValueError as exc:
                raise APIError("Invalid JSON response", response.status_code, attempt) from exc
        raise AssertionError("unreachable")

    def get_pro_matches(self, less_than_match_id: int | None = None) -> list[dict[str, Any]]:
        params = {"less_than_match_id": less_than_match_id} if less_than_match_id else None
        value = self.get_json("proMatches", params)
        if not isinstance(value, list):
            raise APIError("proMatches response is not an array")
        return value

    def get_match_details(self, match_id: int) -> dict[str, Any]:
        value = self.get_json(f"matches/{match_id}")
        if not isinstance(value, dict):
            raise APIError("match response is not an object")
        return value

    def get_heroes(self) -> dict[str, dict[str, Any]]:
        value = self.get_json("constants/heroes")
        if not isinstance(value, dict):
            raise APIError("heroes response is not an object")
        return value

