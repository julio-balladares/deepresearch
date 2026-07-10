from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


@dataclass(slots=True)
class ResearchPlan:
    topic: str
    subquestions: list[str]
    rationale: str = ""


@dataclass(slots=True)
class TargetProfile:
    raw_topic: str
    search_name: str
    identity_hints: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class SearchQueryPlan:
    queries: list[str]


@dataclass(slots=True)
class EvidenceSource:
    title: str
    url: str
    domain: str
    snippet: str
    extracted_text: str
    query: str
    subquestion: str
    fetched: bool = False
    source_type: str = "web"


@dataclass(slots=True)
class GapReport:
    gaps: list[dict[str, Any]]


class ResearchRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    topic: str = Field(..., min_length=1)
    engine: str = Field(default="google", min_length=1)
    max_rounds: int = Field(default=2, ge=1, le=5)
    max_sources: int = Field(default=20, ge=1, le=100)
    max_queries: int = Field(default=8, ge=1, le=100)
    queries_per_subquestion: int = Field(default=2, ge=1, le=10)
    fetch_pages: bool = True
    debug: bool = False


class ResearchResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    topic: str
    target_profile: dict[str, Any]
    created_at: str
    duration_seconds: float
    model: str
    engine: str
    settings: dict[str, Any]
    plan: dict[str, Any]
    query_plans: list[dict[str, Any]]
    rounds: list[dict[str, Any]]
    gap_report: dict[str, Any]
    sources: list[dict[str, Any]]


__all__ = [
    "EvidenceSource",
    "GapReport",
    "ResearchPlan",
    "ResearchRequest",
    "ResearchResult",
    "SearchQueryPlan",
    "TargetProfile",
]
