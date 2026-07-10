from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


BASE_DIR = Path(os.getenv("DEEPRESEARCH_HOME", Path.cwd())).expanduser()

RUNTIME_DIR = BASE_DIR / ".deepresearch"
CACHE_PATH = RUNTIME_DIR / "cache.json"

DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 20
DEFAULT_DOMAIN_LIMIT = 3

DEFAULT_MODEL_NAME = "openai/gpt-4.1-mini"

logger = logging.getLogger("deepresearch")


def env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def env_int(
    name: str,
    default: int,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    raw = env_str(name)

    if not raw:
        return default

    try:
        value = int(raw)
    except ValueError:
        return default

    value = max(minimum, value)

    if maximum is not None:
        value = min(maximum, value)

    return value


def env_bool(name: str, default: bool = False) -> bool:
    raw = env_str(name)

    if not raw:
        return default

    return raw.lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True, slots=True)
class Settings:
    model_name: str = DEFAULT_MODEL_NAME
    llm_provider: str = ""
    api_keys: dict[str, str] = field(default_factory=dict)
    base_urls: dict[str, str] = field(default_factory=dict)
    llm_timeout: int = DEFAULT_TIMEOUT
    serpapi_key: str = ""
    max_rounds: int = 2
    max_sources: int = 20
    max_sources_per_query: int = 5
    max_queries: int = 12
    queries_per_subquestion: int = 2
    max_chars_per_source: int = 6000
    max_final_context_chars: int = 16000
    use_cache: bool = True


def get_settings() -> Settings:
    return Settings(
        model_name=env_str("MODEL_NAME", DEFAULT_MODEL_NAME),
        llm_provider=env_str("LLM_PROVIDER") or env_str("MODEL_PROVIDER"),
        api_keys={
            "anthropic": env_str("ANTHROPIC_API_KEY") or env_str("CLAUDE_API_KEY"),
            "deepseek": env_str("DEEPSEEK_API_KEY"),
            "gemini": env_str("GEMINI_API_KEY") or env_str("GOOGLE_API_KEY"),
            "mistral": env_str("MISTRAL_API_KEY"),
            "ollama": env_str("OLLAMA_API_KEY"),
            "openai": env_str("OPENAI_API_KEY"),
            "openrouter": env_str("OPENROUTER_API_KEY"),
            "perplexity": env_str("PERPLEXITY_API_KEY"),
            "xai": env_str("XAI_API_KEY"),
        },
        base_urls={
            "anthropic": env_str("ANTHROPIC_BASE_URL"),
            "deepseek": env_str("DEEPSEEK_BASE_URL"),
            "gemini": env_str("GEMINI_BASE_URL") or env_str("GOOGLE_AI_BASE_URL"),
            "mistral": env_str("MISTRAL_BASE_URL"),
            "ollama": env_str("OLLAMA_BASE_URL"),
            "openai": env_str("OPENAI_BASE_URL"),
            "openrouter": env_str("OPENROUTER_BASE_URL"),
            "perplexity": env_str("PERPLEXITY_BASE_URL"),
            "xai": env_str("XAI_BASE_URL"),
        },
        llm_timeout=env_int("LLM_TIMEOUT", DEFAULT_TIMEOUT, minimum=1, maximum=300),
        serpapi_key=env_str("SERPAPI_KEY"),
        max_rounds=env_int("MAX_ROUNDS", 2, minimum=1, maximum=5),
        max_sources=env_int("MAX_SOURCES", 20, minimum=3, maximum=100),
        max_sources_per_query=env_int(
            "MAX_SOURCES_PER_QUERY",
            5,
            minimum=1,
            maximum=10,
        ),
        max_queries=env_int("MAX_QUERIES", 12, minimum=1, maximum=100),
        queries_per_subquestion=env_int(
            "QUERIES_PER_SUBQUESTION",
            2,
            minimum=1,
            maximum=10,
        ),
        max_chars_per_source=env_int(
            "MAX_CHARS_PER_SOURCE",
            6000,
            minimum=500,
            maximum=30000,
        ),
        max_final_context_chars=env_int(
            "MAX_FINAL_CONTEXT_CHARS",
            16000,
            minimum=4000,
            maximum=60000,
        ),
        use_cache=env_bool("USE_CACHE", True),
    )


settings = get_settings()


__all__ = [
    "BASE_DIR",
    "CACHE_PATH",
    "DEFAULT_DOMAIN_LIMIT",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TIMEOUT",
    "RUNTIME_DIR",
    "Settings",
    "get_settings",
    "logger",
    "settings",
]
