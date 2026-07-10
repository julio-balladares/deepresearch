from __future__ import annotations

from collections import defaultdict
from typing import Any

from .models import EvidenceSource, GapReport, ResearchPlan


class GapAnalyzer:
    def __init__(self, min_sources_per_subquestion: int = 2) -> None:
        self.min_sources_per_subquestion = max(
            1,
            int(min_sources_per_subquestion),
        )

    def analyze(
        self,
        plan: ResearchPlan,
        evidence: list[EvidenceSource],
    ) -> GapReport:
        by_subquestion: dict[str, list[EvidenceSource]] = defaultdict(list)

        for source in evidence:
            by_subquestion[source.subquestion].append(source)

        gaps: list[dict[str, Any]] = []

        for subquestion in plan.subquestions:
            sources = by_subquestion.get(subquestion, [])

            if len(sources) >= self.min_sources_per_subquestion:
                continue

            gaps.append(
                {
                    "subquestion": subquestion,
                    "reason": (
                        "Insufficient collected evidence "
                        f"({len(sources)}/{self.min_sources_per_subquestion} sources)"
                    ),
                    "sources_found": len(sources),
                    "minimum_required": self.min_sources_per_subquestion,
                }
            )

        return GapReport(gaps=gaps)


__all__ = [
    "GapAnalyzer",
]
