from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
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
        self.path = Path(path)
        self.enabled = bool(enabled)
        self.dirty = False
        self.data: dict[str, Any] = self._empty_cache()
        self._lock = RLock()

        if self.enabled:
            self.load()

    @staticmethod
    def _empty_cache() -> dict[str, Any]:
        return {bucket: {} for bucket in DEFAULT_BUCKETS}

    @staticmethod
    def _is_valid_timestamp(value: Any) -> bool:
        if isinstance(value, bool):
            return False

        if not isinstance(value, (int, float)):
            return False

        return math.isfinite(float(value))

    @staticmethod
    def _is_valid_max_age(value: Any) -> bool:
        if isinstance(value, bool):
            return False

        if not isinstance(value, (int, float)):
            return False

        if not math.isfinite(float(value)):
            return False

        return value >= 0

    def load(self) -> None:
        if not self.enabled:
            return

        with self._lock:
            if not self.path.exists():
                return

            if not self.path.is_file():
                logger.warning(
                    "Cache path exists but is not a file: %s",
                    self.path,
                )
                return

            try:
                with self.path.open("r", encoding="utf-8") as file:
                    loaded = json.load(file)

                if not isinstance(loaded, dict):
                    logger.warning(
                        "Cache root must be a JSON object and will be ignored: %s",
                        self.path,
                    )
                    return

                for bucket in DEFAULT_BUCKETS:
                    entries = loaded.get(bucket, {})

                    if isinstance(entries, dict):
                        self.data[bucket] = entries
                    else:
                        self.data[bucket] = {}

            except json.JSONDecodeError as exc:
                logger.warning(
                    "Cache file is not valid JSON and will be ignored: %s",
                    exc,
                )
            except OSError as exc:
                logger.debug("Could not read cache file: %s", exc)
            except Exception as exc:
                # Cache failures must not interrupt the main workflow.
                logger.debug("Could not load cache: %s", exc)

    def save(self) -> None:
        if not self.enabled or not self.dirty:
            return

        with self._lock:
            if not self.dirty:
                return

            temp_path: Path | None = None

            try:
                ensure_dir(self.path.parent)

                with NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.path.parent,
                    prefix=f".{self.path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temp_file:
                    temp_path = Path(temp_file.name)

                    json.dump(
                        self.data,
                        temp_file,
                        ensure_ascii=False,
                        indent=2,
                    )

                    temp_file.flush()
                    os.fsync(temp_file.fileno())

                os.replace(temp_path, self.path)
                self.dirty = False

            except (OSError, TypeError, ValueError) as exc:
                logger.debug("Could not save cache: %s", exc)
            except Exception as exc:
                # Keep dirty=True so a later save can retry.
                logger.debug("Unexpected error while saving cache: %s", exc)
            finally:
                if temp_path is not None:
                    try:
                        temp_path.unlink(missing_ok=True)
                    except OSError:
                        pass

    def get(self, bucket: str, key: str, max_age_seconds: int) -> Any:
        if not self.enabled:
            return None

        if not bucket or not key:
            return None

        if not self._is_valid_max_age(max_age_seconds):
            logger.debug(
                "Invalid cache max age for bucket %s: %r",
                bucket,
                max_age_seconds,
            )
            return None

        with self._lock:
            entries = self.data.get(bucket)

            if not isinstance(entries, dict):
                return None

            entry = entries.get(key)

            if not isinstance(entry, dict):
                return None

            stored_at = entry.get("stored_at")

            if not self._is_valid_timestamp(stored_at):
                entries.pop(key, None)
                self.dirty = True
                return None

            age_seconds = max(0.0, time.time() - float(stored_at))

            if age_seconds > max_age_seconds:
                entries.pop(key, None)
                self.dirty = True
                return None

            return entry.get("value")

    def set(self, bucket: str, key: str, value: Any) -> None:
        if not self.enabled:
            return

        if not bucket or not key:
            return

        with self._lock:
            entries = self.data.get(bucket)

            if not isinstance(entries, dict):
                entries = {}
                self.data[bucket] = entries

            entries[key] = {
                "stored_at": time.time(),
                "value": value,
            }

            self.dirty = True

    def clear(self) -> None:
        with self._lock:
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
