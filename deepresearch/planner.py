from __future__ import annotations

import json
import re
from typing import Any

from .llm import LLMAgent, LLMRateLimitError
from .models import ResearchPlan, SearchQueryPlan, TargetProfile
from .utils import compact_spaces


MAX_QUERY_LENGTH = 260
MAX_GENERATED_QUERY_LIMIT = 20
ALLOWED_CONTEXT_HINT_TYPES = {"age", "location", "organization", "job_title"}
SEARCH_CONTEXT_HINT_TYPES = {"location", "organization", "job_title"}


class ResearchPlanner:
    def __init__(self, llm: LLMAgent) -> None:
        self.llm = llm

    def generate_plan(self, topic: str) -> ResearchPlan:
        clean_topic = compact_spaces(topic)

        if not clean_topic:
            raise ValueError("A topic is required for planning.")

        return ResearchPlan(
            topic=clean_topic,
            rationale="Analysis plan for reviewing collected public OSINT search results.",
            subquestions=[
                f"Which public profiles appear related to {clean_topic}?",
                f"Which social profiles, portfolios, or public biographies appear for {clean_topic}?",
                f"Which public documents contain relevant mentions of {clean_topic}?",
                "Which results provide ambiguous but potentially relevant identity signals?",
                f"What identity assessment is supported by the evidence for each result related to {clean_topic}?",
            ],
        )

    def generate_search_queries(
        self,
        target_profile: TargetProfile,
        subquestion: str,
        limit: int = 12,
        previous_queries: list[str] | None = None,
        round_number: int = 1,
    ) -> SearchQueryPlan:
        safe_limit = max(1, min(limit, MAX_GENERATED_QUERY_LIMIT))
        safe_round_number = max(1, round_number)
        safe_previous_queries = _normalize_previous_queries(previous_queries)

        prompt = _query_prompt(
            target_profile=target_profile,
            subquestion=subquestion,
            limit=safe_limit,
            previous_queries=safe_previous_queries,
            round_number=safe_round_number,
        )

        try:
            response = self.llm.ask(
                prompt,
                system_prompt=_query_system_prompt(),
                temperature=0.1,
                max_tokens=700,
                json_mode=True,
            )
            generated_queries = _parse_query_response(response)
        except LLMRateLimitError:
            raise
        except Exception as exc:
            raise RuntimeError(f"LLM query generation failed: {exc}") from exc

        queries = _build_query_strategy(
            target_profile=target_profile,
            generated_queries=generated_queries,
            limit=safe_limit,
            previous_queries=safe_previous_queries,
        )

        if not queries:
            raise ValueError("LLM query generation returned no usable queries.")

        return SearchQueryPlan(queries=queries)

    def parse_target(self, topic: str) -> TargetProfile:
        raw_topic = compact_spaces(topic)

        if not raw_topic:
            raise ValueError("A topic is required for target interpretation.")

        try:
            response = self.llm.ask(
                _target_prompt(raw_topic),
                system_prompt=_target_system_prompt(),
                temperature=0.0,
                max_tokens=500,
                json_mode=True,
            )
        except LLMRateLimitError:
            raise
        except Exception as exc:
            raise RuntimeError(f"LLM target interpretation failed: {exc}") from exc

        return _parse_target_response(raw_topic, response)


def _target_system_prompt() -> str:
    return (
        "You interpret raw OSINT target input into structured English fields. "
        "Return strict JSON only."
    )


def _target_prompt(raw_topic: str) -> str:
    return f"""
Interpret this raw target input for an OSINT collection pipeline:
{raw_topic}

Extract the target name that should be used for public search matching.
Move extra user-provided facts into identity_hints.
Normalize every hint type and label to English.
Do not invent facts, names, identifiers, usernames, locations, or organizations.
If a value is ambiguous, keep it as an identity_hints item with type "identifier".

Allowed hint types include:
age, passport, dni, national_id, identity_card, student_id, license, drivers_license,
location, organization, job_title, username, alias, domain, email, document, identifier.

Return JSON with this exact shape:
{{
  "search_name": "person or target name only",
  "identity_hints": [
    {{"type": "age", "label": "age", "value": "22", "source": "user_input"}}
  ]
}}
""".strip()


def _query_system_prompt() -> str:
    return (
        "You generate high-quality public web search queries for OSINT result collection. "
        "You may use advanced search operators such as site:, filetype:, intitle:, and inurl: "
        "when they improve precision or help collect distinct public evidence. "
        "Return strict JSON only."
    )


def _query_prompt(
    target_profile: TargetProfile,
    subquestion: str,
    limit: int,
    previous_queries: list[str],
    round_number: int,
) -> str:
    hints_allowed_in_search = [
        hint
        for hint in target_profile.identity_hints
        if hint.get("type") in ALLOWED_CONTEXT_HINT_TYPES
    ]

    return f"""
Create up to {limit} prioritized public web search queries for collecting already-public results.

Research round: {round_number}
Raw user input: {target_profile.raw_topic}
Target name for matching: {target_profile.search_name}
Comparison hints available for analysis: {json.dumps(target_profile.identity_hints, ensure_ascii=False)}
Hints allowed in search terms: {json.dumps(hints_allowed_in_search, ensure_ascii=False)}
Analysis vector: {subquestion}
Queries already executed: {json.dumps(previous_queries, ensure_ascii=False)}

The queries should follow a conservative broad-to-specific OSINT search strategy.
For round 1, start with the exact target name and then the unquoted full name.
For later rounds, focus only on unresolved analysis vectors and use different query wording.
Never return a query listed under Queries already executed.
Use user-provided location or organization context before guessing a platform.
Use advanced search operators when they add clear investigative value.
Useful operators include site:, filetype:, intitle:, and inurl:.
Generate dorks for public profiles, documents, biographies, portfolios, mentions, and indexable public pages.
Do not generate one query per social network unless the platform is strongly justified by the analysis vector.
Do not use a platform filter merely because the platform exists.
Avoid speculative usernames, aliases, locations, organizations, and name variants.
Order all remaining queries by expected relevance and usefulness for the analysis vector.
Do not treat raw identity document numbers as the person's name.

Return JSON with this shape:
{{"queries":["query 1","query 2"]}}
""".strip()


def _parse_target_response(raw_topic: str, response: str) -> TargetProfile:
    data = _extract_json_object(response)

    search_name = compact_spaces(str(data.get("search_name", "")))

    if not search_name:
        raise ValueError("LLM target interpretation returned no search_name.")

    raw_hints = data.get("identity_hints", [])
    identity_hints: list[dict[str, str]] = []

    if isinstance(raw_hints, list):
        for item in raw_hints:
            if not isinstance(item, dict):
                continue

            hint_type = _normalize_hint_label(
                str(item.get("type") or item.get("label") or "identifier")
            )

            value = compact_spaces(str(item.get("value", "")))

            if not value:
                continue

            identity_hints.append(
                {
                    "type": hint_type,
                    "label": hint_type,
                    "value": value,
                    "source": "user_input",
                }
            )

    return TargetProfile(
        raw_topic=raw_topic,
        search_name=search_name,
        identity_hints=_unique_hints(identity_hints),
    )


def _parse_query_response(response: str) -> list[str]:
    data = _extract_json_object(response)
    queries = data.get("queries", [])

    if not isinstance(queries, list):
        return []

    return [str(query) for query in queries if isinstance(query, str)]


def _build_query_strategy(
    target_profile: TargetProfile,
    generated_queries: list[str],
    limit: int,
    previous_queries: list[str] | None = None,
) -> list[str]:
    search_name = compact_spaces(target_profile.search_name)

    if not search_name:
        return []

    exact_query = json.dumps(search_name, ensure_ascii=False)
    previous_keys = {
        compact_spaces(query).casefold()
        for query in (previous_queries or [])
        if compact_spaces(query)
    }

    candidates: list[str] = []

    if exact_query.casefold() not in previous_keys:
        candidates.append(exact_query)

    if search_name.casefold() not in previous_keys:
        candidates.append(search_name)

    contextual_hints = [
        compact_spaces(str(hint.get("value", "")))
        for hint in target_profile.identity_hints
        if hint.get("type") in SEARCH_CONTEXT_HINT_TYPES
        and compact_spaces(str(hint.get("value", "")))
    ]

    for hint in contextual_hints[:1]:
        candidates.append(f"{exact_query} {json.dumps(hint, ensure_ascii=False)}")

    cleaned_generated = _clean_queries(generated_queries, max(limit * 3, limit))

    for query in cleaned_generated:
        query_key = query.casefold()

        if query_key in {exact_query.casefold(), search_name.casefold()}:
            continue

        candidates.append(query)

    return [
        query
        for query in _clean_queries(candidates, max(limit * 2, limit))
        if query.casefold() not in previous_keys
    ][:limit]


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start < 0 or end <= start:
            return {}

        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return {}

    return parsed if isinstance(parsed, dict) else {}


def _clean_queries(queries: list[str], limit: int) -> list[str]:
    cleaned_queries: list[str] = []
    seen: set[str] = set()

    for query in queries:
        query = compact_spaces(query)
        query = query.replace("“", '"').replace("”", '"')

        if not _query_is_usable(query):
            continue

        key = query.casefold()

        if key in seen:
            continue

        seen.add(key)
        cleaned_queries.append(query)

        if len(cleaned_queries) >= limit:
            break

    return cleaned_queries


def _query_is_usable(query: str) -> bool:
    if not query:
        return False

    if len(query) > MAX_QUERY_LENGTH:
        return False

    if "\n" in query or "\r" in query:
        return False

    return True


def _normalize_hint_label(label: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", label.lower()).strip("_")
    return cleaned or "identifier"


def _normalize_previous_queries(previous_queries: list[str] | None) -> list[str]:
    if not previous_queries:
        return []

    normalized: list[str] = []

    for query in previous_queries:
        cleaned = compact_spaces(query)

        if cleaned:
            normalized.append(cleaned)

    return normalized


def _unique_hints(hints: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []

    for hint in hints:
        hint_type = compact_spaces(hint.get("type", "")) or "identifier"
        value = compact_spaces(hint.get("value", ""))

        if not value:
            continue

        key = (hint_type.casefold(), value.casefold())

        if key in seen:
            continue

        seen.add(key)

        unique.append(
            {
                "type": hint_type,
                "label": compact_spaces(hint.get("label", "")) or hint_type,
                "value": value,
                "source": compact_spaces(hint.get("source", "")) or "user_input",
            }
        )

    return unique


__all__ = ["ResearchPlanner"]
