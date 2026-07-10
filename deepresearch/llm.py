from __future__ import annotations

import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import requests

from .cache import LLM_CACHE_BUCKET, LLM_CACHE_TTL_SECONDS, LocalCache
from .config import DEFAULT_MAX_TOKENS, Settings, logger
from .utils import compact_spaces


MAX_RATE_LIMIT_WAIT_SECONDS = 60.0
MAX_LLM_RETRIES = 2

ChatMessage = dict[str, str]


class LLMError(RuntimeError):
    pass


class LLMConfigurationError(LLMError):
    pass


class LLMProviderError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str
    usage: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ProviderSelection:
    provider: str
    model: str


class LLMProvider(ABC):
    name: str

    def __init__(
        self,
        api_key: str,
        timeout_seconds: int,
        base_url: str = "",
    ) -> None:
        self.api_key = compact_spaces(api_key)
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.base_url = compact_spaces(base_url)

    @abstractmethod
    def create_completion(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        raise NotImplementedError

    def _require_api_key(self, env_name: str) -> None:
        if not self.api_key:
            raise LLMConfigurationError(f"{env_name} is missing from the environment.")

    def _post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(1, MAX_LLM_RETRIES + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )

                if response.status_code == 429:
                    wait_seconds = _retry_after_seconds(response)

                    if (
                        attempt < MAX_LLM_RETRIES
                        and 0 <= wait_seconds <= MAX_RATE_LIMIT_WAIT_SECONDS
                    ):
                        sleep_for = wait_seconds + 1
                        logger.warning(
                            "%s rate limit reached; retrying in %.1fs | attempt=%s/%s",
                            self.name,
                            sleep_for,
                            attempt,
                            MAX_LLM_RETRIES,
                        )
                        time.sleep(sleep_for)
                        continue

                    raise LLMRateLimitError(_response_error_message(response))

                if response.status_code >= 400:
                    raise LLMProviderError(_response_error_message(response))

                data = response.json()

                if not isinstance(data, dict):
                    raise LLMProviderError(f"{self.name} returned a non-object response.")

                return data

            except LLMError:
                raise

            except requests.Timeout as exc:
                last_error = exc

                if attempt >= MAX_LLM_RETRIES:
                    break

                sleep_for = 1.5**attempt
                logger.warning(
                    "%s timeout; retrying in %.1fs | attempt=%s/%s",
                    self.name,
                    sleep_for,
                    attempt,
                    MAX_LLM_RETRIES,
                )
                time.sleep(sleep_for)

            except requests.RequestException as exc:
                last_error = exc

                if attempt >= MAX_LLM_RETRIES:
                    break

                sleep_for = 1.5**attempt
                logger.warning(
                    "%s connection error; retrying in %.1fs | attempt=%s/%s | error=%s",
                    self.name,
                    sleep_for,
                    attempt,
                    MAX_LLM_RETRIES,
                    exc,
                )
                time.sleep(sleep_for)

            except ValueError as exc:
                raise LLMProviderError(f"{self.name} returned invalid JSON.") from exc

        raise LLMProviderError(
            f"{self.name} request failed after {MAX_LLM_RETRIES} attempts: {last_error}"
        ) from last_error


class OpenAICompatibleProvider(LLMProvider):
    api_key_env = ""
    default_base_url = ""
    name = "openai-compatible"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: int,
        base_url: str = "",
    ) -> None:
        super().__init__(
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            base_url=base_url or self.default_base_url,
        )
        self._require_api_key(self.api_key_env)

    def create_completion(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = self._post_json(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
        )

        choices = data.get("choices")

        if not isinstance(choices, list) or not choices:
            raise LLMProviderError(f"{self.name} returned no choices.")

        first_choice = choices[0]

        if not isinstance(first_choice, dict):
            raise LLMProviderError(f"{self.name} returned an invalid choice.")

        message = first_choice.get("message", {})

        if not isinstance(message, dict):
            raise LLMProviderError(f"{self.name} returned no message.")

        content = str(message.get("content") or "").strip()

        if not content:
            finish_reason = first_choice.get("finish_reason", "unknown")
            raise LLMProviderError(
                f"{self.name} returned an empty answer. finish_reason={finish_reason}"
            )

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        return LLMResponse(content=content, usage=usage)


class OpenAIProvider(OpenAICompatibleProvider):
    api_key_env = "OPENAI_API_KEY"
    default_base_url = "https://api.openai.com/v1"
    name = "openai"


class MistralProvider(OpenAICompatibleProvider):
    api_key_env = "MISTRAL_API_KEY"
    default_base_url = "https://api.mistral.ai/v1"
    name = "mistral"


class DeepSeekProvider(OpenAICompatibleProvider):
    api_key_env = "DEEPSEEK_API_KEY"
    default_base_url = "https://api.deepseek.com"
    name = "deepseek"


class XAIProvider(OpenAICompatibleProvider):
    api_key_env = "XAI_API_KEY"
    default_base_url = "https://api.x.ai/v1"
    name = "xai"


class OpenRouterProvider(OpenAICompatibleProvider):
    api_key_env = "OPENROUTER_API_KEY"
    default_base_url = "https://openrouter.ai/api/v1"
    name = "openrouter"


class PerplexityProvider(OpenAICompatibleProvider):
    api_key_env = "PERPLEXITY_API_KEY"
    default_base_url = "https://api.perplexity.ai"
    name = "perplexity"


class OllamaProvider(OpenAICompatibleProvider):
    api_key_env = ""
    default_base_url = "http://localhost:11434/v1"
    name = "ollama"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: int,
        base_url: str = "",
    ) -> None:
        LLMProvider.__init__(
            self,
            api_key=api_key or "ollama",
            timeout_seconds=timeout_seconds,
            base_url=base_url or self.default_base_url,
        )


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: int,
        base_url: str = "",
    ) -> None:
        super().__init__(
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            base_url=base_url or "https://api.anthropic.com",
        )
        self._require_api_key("ANTHROPIC_API_KEY")

    def create_completion(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        system_prompt = _join_system_messages(messages)
        user_messages = [
            message for message in messages if message.get("role") != "system"
        ]

        if json_mode:
            system_prompt = compact_spaces(
                f"{system_prompt} Return strict valid JSON only."
            )

        payload: dict[str, Any] = {
            "model": model,
            "messages": user_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system_prompt:
            payload["system"] = system_prompt

        data = self._post_json(
            url=f"{self.base_url.rstrip('/')}/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            payload=payload,
        )

        content_blocks = data.get("content", [])
        content = _extract_text_blocks(content_blocks)

        if not content:
            stop_reason = data.get("stop_reason", "unknown")
            raise LLMProviderError(
                f"{self.name} returned an empty answer. stop_reason={stop_reason}"
            )

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        return LLMResponse(content=content, usage=usage)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: int,
        base_url: str = "",
    ) -> None:
        super().__init__(
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            base_url=base_url or "https://generativelanguage.googleapis.com/v1beta",
        )
        self._require_api_key("GEMINI_API_KEY")

    def create_completion(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        system_prompt = _join_system_messages(messages)
        payload: dict[str, Any] = {
            "contents": _gemini_contents(messages),
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}],
            }

        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        data = self._post_json(
            url=(
                f"{self.base_url.rstrip('/')}/models/"
                f"{model}:generateContent?key={self.api_key}"
            ),
            headers={"Content-Type": "application/json"},
            payload=payload,
        )

        candidates = data.get("candidates")

        if not isinstance(candidates, list) or not candidates:
            raise LLMProviderError(f"{self.name} returned no candidates.")

        first_candidate = candidates[0]

        if not isinstance(first_candidate, dict):
            raise LLMProviderError(f"{self.name} returned an invalid candidate.")

        content = _gemini_candidate_text(first_candidate)

        if not content:
            finish_reason = first_candidate.get("finishReason", "unknown")
            raise LLMProviderError(
                f"{self.name} returned an empty answer. finish_reason={finish_reason}"
            )

        usage = (
            data.get("usageMetadata")
            if isinstance(data.get("usageMetadata"), dict)
            else None
        )
        return LLMResponse(content=content, usage=usage)


PROVIDER_CLASSES: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "claude": AnthropicProvider,
    "deepseek": DeepSeekProvider,
    "gemini": GeminiProvider,
    "google": GeminiProvider,
    "gpt": OpenAIProvider,
    "mistral": MistralProvider,
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "openrouter": OpenRouterProvider,
    "perplexity": PerplexityProvider,
    "xai": XAIProvider,
}

PROVIDER_NAMES = set(PROVIDER_CLASSES)


class LLMAgent:
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        timeout_seconds: int,
        base_url: str = "",
        cache: LocalCache | None = None,
    ) -> None:
        self.provider_name = _normalize_provider(provider)
        self.model = compact_spaces(model)
        self.cache = cache

        if not self.model:
            raise LLMConfigurationError("LLM model name is missing.")

        provider_class = PROVIDER_CLASSES.get(self.provider_name)

        if provider_class is None:
            supported = ", ".join(sorted(PROVIDER_NAMES))
            raise LLMConfigurationError(
                f"Unsupported LLM provider '{provider}'. Supported providers: {supported}."
            )

        self.provider = provider_class(
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            base_url=base_url,
        )

    @property
    def model_identifier(self) -> str:
        return f"{self.provider_name}/{self.model}"

    def ask(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        json_mode: bool = False,
    ) -> str:
        clean_prompt = compact_spaces(prompt)
        clean_system_prompt = compact_spaces(system_prompt) if system_prompt else None
        safe_temperature = max(0.0, min(float(temperature), 2.0))
        safe_max_tokens = max(1, int(max_tokens))

        if not clean_prompt:
            raise ValueError("The prompt cannot be empty.")

        cache_key = self._cache_key(
            prompt=clean_prompt,
            system_prompt=clean_system_prompt,
            temperature=safe_temperature,
            max_tokens=safe_max_tokens,
            json_mode=json_mode,
        )

        cached = self._get_cached_response(cache_key)

        if cached:
            return cached

        messages: list[ChatMessage] = []

        if clean_system_prompt:
            messages.append({"role": "system", "content": clean_system_prompt})

        messages.append({"role": "user", "content": clean_prompt})

        logger.info(
            "LLM request | provider=%s | model=%s | chars=%s | json_mode=%s",
            self.provider_name,
            self.model,
            len(clean_prompt),
            json_mode,
        )

        response = self.provider.create_completion(
            model=self.model,
            messages=messages,
            temperature=safe_temperature,
            max_tokens=safe_max_tokens,
            json_mode=json_mode,
        )

        self._log_usage(response.usage)
        self._set_cached_response(cache_key, response.content)

        return response.content

    def _get_cached_response(self, cache_key: str) -> str | None:
        if self.cache is None:
            return None

        cached = self.cache.get(
            LLM_CACHE_BUCKET,
            cache_key,
            LLM_CACHE_TTL_SECONDS,
        )

        if isinstance(cached, str) and cached:
            logger.info("LLM cache hit | model=%s", self.model_identifier)
            return cached

        return None

    def _set_cached_response(self, cache_key: str, content: str) -> None:
        if self.cache is None:
            return

        self.cache.set(LLM_CACHE_BUCKET, cache_key, content)

    def _cache_key(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        payload = {
            "model": self.model_identifier,
            "system_prompt": system_prompt or "",
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "json_mode": json_mode,
        }

        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _log_usage(usage: dict[str, Any] | None) -> None:
        if not usage:
            return

        logger.info("LLM usage | %s", usage)


def build_llm_agent(settings: Settings, cache: LocalCache | None = None) -> LLMAgent:
    selection = select_provider(
        provider=settings.llm_provider,
        model=settings.model_name,
    )

    return LLMAgent(
        provider=selection.provider,
        model=selection.model,
        api_key=settings.api_keys.get(selection.provider, ""),
        timeout_seconds=settings.llm_timeout,
        base_url=settings.base_urls.get(selection.provider, ""),
        cache=cache,
    )


def select_provider(provider: str, model: str) -> ProviderSelection:
    clean_provider = _normalize_provider(provider)
    clean_model = compact_spaces(model)

    if not clean_model:
        raise LLMConfigurationError("LLM model name is missing.")

    prefix, _, model_without_prefix = clean_model.partition("/")
    normalized_prefix = _normalize_provider(prefix)

    if not clean_provider and normalized_prefix in PROVIDER_NAMES and model_without_prefix:
        clean_provider = normalized_prefix
        clean_model = model_without_prefix
    elif clean_provider and normalized_prefix == clean_provider and model_without_prefix:
        clean_model = model_without_prefix
    elif not clean_provider:
        clean_provider = "openai"

    return ProviderSelection(provider=clean_provider, model=clean_model)


def _normalize_provider(provider: str) -> str:
    cleaned = compact_spaces(provider).lower()

    if cleaned == "claude":
        return "anthropic"

    if cleaned == "google":
        return "gemini"

    if cleaned == "gpt":
        return "openai"

    return cleaned


def _join_system_messages(messages: list[ChatMessage]) -> str:
    return "\n".join(
        message["content"]
        for message in messages
        if message.get("role") == "system" and compact_spaces(message.get("content"))
    )


def _extract_text_blocks(blocks: Any) -> str:
    if not isinstance(blocks, list):
        return ""

    parts: list[str] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue

        if block.get("type") == "text":
            text = compact_spaces(block.get("text", ""))

            if text:
                parts.append(text)

    return "\n".join(parts).strip()


def _gemini_contents(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role")

        if role == "system":
            continue

        gemini_role = "model" if role == "assistant" else "user"
        content = compact_spaces(message.get("content", ""))

        if content:
            contents.append(
                {
                    "role": gemini_role,
                    "parts": [{"text": content}],
                }
            )

    return contents


def _gemini_candidate_text(candidate: dict[str, Any]) -> str:
    content = candidate.get("content", {})

    if not isinstance(content, dict):
        return ""

    parts = content.get("parts", [])

    if not isinstance(parts, list):
        return ""

    text_parts = [
        compact_spaces(part.get("text", ""))
        for part in parts
        if isinstance(part, dict) and compact_spaces(part.get("text", ""))
    ]

    return "\n".join(text_parts).strip()


def _response_error_message(response: requests.Response) -> str:
    fallback = f"HTTP {response.status_code}: {response.text[:500]}"

    try:
        data = response.json()
    except ValueError:
        return compact_spaces(fallback)

    if not isinstance(data, dict):
        return compact_spaces(fallback)

    error = data.get("error", data)

    if isinstance(error, dict):
        return compact_spaces(
            error.get("message")
            or error.get("error")
            or error.get("detail")
            or fallback
        )

    if isinstance(error, str):
        return compact_spaces(error)

    return compact_spaces(fallback)


def _retry_after_seconds(response: requests.Response) -> float:
    retry_after = response.headers.get("retry-after")

    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass

    message = _response_error_message(response)
    match = re.search(
        r"try again in\s+(?:(?P<minutes>\d+(?:\.\d+)?)m)?"
        r"(?P<seconds>\d+(?:\.\d+)?)s",
        message,
        re.IGNORECASE,
    )

    if not match:
        return -1.0

    minutes = float(match.group("minutes") or 0)
    seconds = float(match.group("seconds"))

    return minutes * 60 + seconds


__all__ = [
    "LLMAgent",
    "LLMConfigurationError",
    "LLMError",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMResponse",
    "ProviderSelection",
    "build_llm_agent",
    "select_provider",
]
