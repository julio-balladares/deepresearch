from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .config import logger
from .utils import ensure_dir


SEARCH_CACHE_BUCKET = "search"
PAGES_CACHE_BUCKET = "pages"
LLM_CACHE_BUCKET = "llm"

SEARCH_CACHE_TTL_SECONDS = 21_600
PAGE_CACHE_TTL_SECONDS = 86_400
LLM_CACHE_TTL_SECONDS = 86_400

DEFAULT_BUCKETS = (
    SEARCH_CACHE_BUCKET,
    PAGES_CACHE_BUCKET,
    LLM_CACHE_BUCKET,
)


class LocalCache:
    def __init__(self, path: Path, enabled: bool = True) -> None:
        self.path = path
        self.enabled = bool(enabled)
        self.dirty = False
        self.data: dict[str, Any] = self._empty_cache()

        if self.enabled:
            self.load()

    @staticmethod
    def _empty_cache() -> dict[str, Any]:
        return {bucket: {} for bucket in DEFAULT_BUCKETS}

    def load(self) -> None:
        if not self.enabled:
            return

        if not self.path.exists():
            return

        try:
            with self.path.open("r", encoding="utf-8") as file:
                loaded = json.load(file)

            if not isinstance(loaded, dict):
                return

            for bucket in DEFAULT_BUCKETS:
                entries = loaded.get(bucket, {})

                if isinstance(entries, dict):
                    self.data[bucket] = entries

        except json.JSONDecodeError as exc:
            logger.warning("Cache file is not valid JSON and will be ignored: %s", exc)
        except OSError as exc:
            logger.debug("Could not read cache file: %s", exc)
        except Exception as exc:
            logger.debug("Could not load cache: %s", exc)

    def save(self) -> None:
        if not self.enabled:
            return

        if not self.dirty:
            return

        ensure_dir(self.path.parent)

        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")

        try:
            with temp_path.open("w", encoding="utf-8") as file:
                json.dump(self.data, file, ensure_ascii=False, indent=2)

            os.replace(temp_path, self.path)
            self.dirty = False

        except Exception as exc:
            logger.debug("Could not save cache: %s", exc)

            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass

    def get(self, bucket: str, key: str, max_age_seconds: int) -> Any:
        if not self.enabled:
            return None

        if not bucket or not key:
            return None

        entries = self.data.get(bucket)

        if not isinstance(entries, dict):
            return None

        entry = entries.get(key)

        if not isinstance(entry, dict):
            return None

        stored_at = entry.get("stored_at")

        if not isinstance(stored_at, (int, float)):
            entries.pop(key, None)
            self.dirty = True
            return None

        if time.time() - stored_at > max_age_seconds:
            entries.pop(key, None)
            self.dirty = True
            return None

        return entry.get("value")

    def set(self, bucket: str, key: str, value: Any) -> None:
        if not self.enabled:
            return

        if not bucket or not key:
            return

        entries = self.data.setdefault(bucket, {})

        if not isinstance(entries, dict):
            entries = {}
            self.data[bucket] = entries

        entries[key] = {
            "stored_at": time.time(),
            "value": value,
        }

        self.dirty = True

    def clear(self) -> None:
        self.data = self._empty_cache()
        self.dirty = True
        self.save()


__all__ = [
    "LLM_CACHE_BUCKET",
    "LLM_CACHE_TTL_SECONDS",
    "LocalCache",
    "PAGE_CACHE_TTL_SECONDS",
    "PAGES_CACHE_BUCKET",
    "SEARCH_CACHE_BUCKET",
    "SEARCH_CACHE_TTL_SECONDS",
]
