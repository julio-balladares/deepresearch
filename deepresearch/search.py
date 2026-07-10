from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import requests

from .cache import LocalCache, SEARCH_CACHE_BUCKET, SEARCH_CACHE_TTL_SECONDS
from .config import DEFAULT_TIMEOUT, logger
from .utils import compact_spaces, domain_from_url


SERPAPI_URL = "https://serpapi.com/search.json"
MAX_SERPAPI_RESULTS_PER_QUERY = 10


class SearchError(RuntimeError):
    pass


class SearchConfigurationError(SearchError):
    pass


class SearchProviderError(SearchError):
    pass


class SearchAgent:
    SUPPORTED_ENGINES = {"google", "bing", "yahoo", "duckduckgo", "yandex"}

    def __init__(
        self,
        serpapi_key: str,
        engine: str,
        cache: LocalCache,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
    ) -> None:
        self.serpapi_key = compact_spaces(serpapi_key)
        self.engine = compact_spaces(engine).lower()
        self.cache = cache
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff = max(0.1, float(retry_backoff))

        if not self.serpapi_key:
            raise SearchConfigurationError(
                "SERPAPI_KEY is missing from the environment."
            )

        if self.engine not in self.SUPPORTED_ENGINES:
            raise SearchConfigurationError(f"Unsupported search engine: {self.engine}")

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(compatible; DeepResearch/1.0; +https://serpapi.com/)"
                )
            }
        )

    def close(self) -> None:
        self.session.close()

    def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        clean_query = compact_spaces(query)

        if not clean_query:
            raise ValueError("Search query cannot be empty.")

        result_count = max(1, min(int(max_results), MAX_SERPAPI_RESULTS_PER_QUERY))

        logger.info("OSINT search | engine=%s | query=%s", self.engine, clean_query)

        params = self._search_params(clean_query, result_count)
        data = self._serpapi_request(params)

        organic_results = data.get("organic_results", []) or []

        if not isinstance(organic_results, list):
            return []

        results: list[dict[str, str]] = []

        for item in organic_results:
            if not isinstance(item, dict):
                continue

            url = compact_spaces(item.get("link", ""))

            if not url:
                continue

            domain = domain_from_url(url) or compact_spaces(
                item.get("displayed_link", "")
            )

            results.append(
                {
                    "title": compact_spaces(item.get("title", "")),
                    "url": url,
                    "snippet": compact_spaces(item.get("snippet", "")),
                    "domain": domain,
                }
            )

        return results

    def _serpapi_request(self, params: dict[str, Any]) -> dict[str, Any]:
        request_params = dict(params)
        request_params["api_key"] = self.serpapi_key
        request_params["engine"] = self.engine

        cache_key = self._cache_key(request_params)

        cached = self.cache.get(
            SEARCH_CACHE_BUCKET,
            cache_key,
            SEARCH_CACHE_TTL_SECONDS,
        )

        if isinstance(cached, dict):
            logger.debug("Search cache hit")
            return cached

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(
                    SERPAPI_URL,
                    params=request_params,
                    timeout=DEFAULT_TIMEOUT,
                )
                response.raise_for_status()

                data = response.json()

                if not isinstance(data, dict):
                    raise SearchProviderError("SerpApi returned a non-object response.")

                if self._is_no_results_response(data):
                    logger.info("OSINT search returned no results")
                    self.cache.set(SEARCH_CACHE_BUCKET, cache_key, data)
                    return data

                self._raise_for_serpapi_error(data)

                self.cache.set(SEARCH_CACHE_BUCKET, cache_key, data)
                return data

            except requests.exceptions.RequestException as exc:
                last_error = exc

                if attempt >= self.max_retries:
                    break

                sleep_for = self.retry_backoff**attempt
                logger.warning(
                    "SerpApi request failed; retrying in %.1fs | attempt=%s/%s | error=%s",
                    sleep_for,
                    attempt,
                    self.max_retries,
                    exc,
                )
                time.sleep(sleep_for)

            except ValueError as exc:
                raise SearchProviderError(
                    f"SerpApi returned invalid JSON: {exc}"
                ) from exc

        raise SearchProviderError(
            f"SerpApi request failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def _search_params(self, query: str, result_count: int) -> dict[str, Any]:
        if self.engine == "google":
            return {
                "q": query,
                "num": result_count,
            }

        if self.engine == "bing":
            return {
                "q": query,
                "count": result_count,
            }

        if self.engine == "yahoo":
            return {
                "p": query,
            }

        if self.engine == "duckduckgo":
            return {
                "q": query,
            }

        if self.engine == "yandex":
            return {
                "text": query,
                "groups_on_page": result_count,
            }

        raise SearchConfigurationError(f"Unsupported search engine: {self.engine}")

    @staticmethod
    def _cache_key(params: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(
                params,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _raise_for_serpapi_error(data: dict[str, Any]) -> None:
        error = data.get("error")

        if error:
            raise SearchProviderError(f"SerpApi error: {error}")

        metadata = data.get("search_metadata", {})

        if (
            isinstance(metadata, dict)
            and str(metadata.get("status", "")).lower() == "error"
        ):
            raise SearchProviderError("SerpApi search metadata status is Error.")

    @staticmethod
    def _is_no_results_response(data: dict[str, Any]) -> bool:
        error = compact_spaces(data.get("error", "")).lower()

        no_result_messages = (
            "hasn't returned any results",
            "has not returned any results",
            "no results for this query",
            "did not return any results",
        )

        if error and any(message in error for message in no_result_messages):
            return True

        search_information = data.get("search_information", {})

        if not isinstance(search_information, dict):
            return False

        total_results = search_information.get("total_results")

        return total_results == 0 and not data.get("organic_results")


__all__ = [
    "SearchAgent",
    "SearchConfigurationError",
    "SearchError",
    "SearchProviderError",
]
