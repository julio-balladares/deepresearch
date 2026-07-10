from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def setup_logging(debug: bool = False) -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)

        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compact_spaces(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def domain_from_url(url: Any) -> str:
    parsed = urlparse(str(url or "").strip())
    domain = parsed.netloc.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def slugify(text: Any, max_length: int = 60) -> str:
    cleaned = compact_spaces(text).lower()
    cleaned = re.sub(r"[\s-]+", "_", cleaned)
    cleaned = re.sub(r"[^\w_]+", "", cleaned)
    cleaned = cleaned.strip("_")

    return cleaned[:max_length] if cleaned else "untitled"


def truncate_text(text: Any, max_chars: int) -> str:
    clean_text = str(text or "")

    if max_chars <= 0:
        return ""

    if len(clean_text) <= max_chars:
        return clean_text

    cut = clean_text[:max_chars]
    last_sentence = max(cut.rfind(". "), cut.rfind("\n"))

    if last_sentence > max_chars * 0.65:
        cut = cut[: last_sentence + 1]

    return cut.rstrip() + "\n...[TRUNCATED]"


__all__ = [
    "compact_spaces",
    "domain_from_url",
    "ensure_dir",
    "setup_logging",
    "slugify",
    "truncate_text",
]
