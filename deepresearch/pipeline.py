from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Callable

from .cache import LocalCache
from .config import CACHE_PATH, DEFAULT_DOMAIN_LIMIT, Settings, logger, settings
from .gaps import GapAnalyzer
from .llm import build_llm_agent
from .models import EvidenceSource, GapReport, ResearchRequest, ResearchResult
from .planner import ResearchPlanner
from .reader import SourceReader
from .search import SearchAgent
from .utils import compact_spaces, domain_from_url, setup_logging

ProgressCallback = Callable[[str, dict[str, Any]], None]


class DeepResearchPipeline:
    def __init__(
        self,
        config: Settings | None = None,
        engine: str = "google",
        debug: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        if debug:
            setup_logging(debug=True)

        self.config = config or settings
        self.engine = compact_spaces(engine).lower() or "google"
        self.progress_callback = progress_callback
        self.cache = LocalCache(CACHE_PATH, enabled=self.config.use_cache)

        self.llm = build_llm_agent(self.config, cache=self.cache)

        self.planner = ResearchPlanner(self.llm)

        self.search_agent = SearchAgent(
            self.config.serpapi_key,
            engine=self.engine,
            cache=self.cache,
        )

        self.reader = SourceReader(
            session=self.search_agent.session,
            cache=self.cache,
            max_chars_per_source=self.config.max_chars_per_source,
        )

        self.gap_analyzer = GapAnalyzer()

    def close(self) -> None:
        self.search_agent.close()
        self.cache.save()

    def _emit(self, event: str, payload: dict[str, Any] | None = None) -> None:
        if self.progress_callback is not None:
            self.progress_callback(event, payload or {})

    def run(self, request: ResearchRequest) -> ResearchResult:
        topic = compact_spaces(request.topic)

        if not topic:
            raise ValueError("A topic is required for analysis.")

        max_rounds = max(1, min(request.max_rounds or self.config.max_rounds, 5))
        max_sources = max(1, min(request.max_sources or self.config.max_sources, 100))
        max_queries = max(1, min(request.max_queries or self.config.max_queries, 100))
        queries_per_subquestion = max(
            1,
            min(
                request.queries_per_subquestion or self.config.queries_per_subquestion,
                10,
            ),
        )

        started = time.time()

        try:
            self._emit(
                "started",
                {
                    "topic": topic,
                    "engine": self.engine,
                    "max_rounds": max_rounds,
                    "max_sources": max_sources,
                    "max_queries": max_queries,
                },
            )

            target_profile = self.planner.parse_target(topic)
            search_topic = target_profile.search_name
            plan = self.planner.generate_plan(search_topic)

            self._emit(
                "plan_generated",
                {
                    "topic": topic,
                    "search_name": search_topic,
                    "analysis_vectors": len(plan.subquestions),
                    "subquestions": plan.subquestions,
                },
            )

            all_sources: list[EvidenceSource] = []
            seen_urls: set[str] = set()
            domain_counts: dict[str, int] = defaultdict(int)
            completed_queries: set[str] = set()
            round_summaries: list[dict[str, Any]] = []
            query_plans: list[dict[str, Any]] = []
            gap_report = GapReport(gaps=[])

            target_subquestions = plan.subquestions[:]

            logger.info(
                "DeepResearch pipeline started | topic=%s | search_name=%s | engine=%s",
                topic,
                search_topic,
                self.engine,
            )

            for round_number in range(1, max_rounds + 1):
                remaining_query_budget = max_queries - len(completed_queries)

                if remaining_query_budget <= 0:
                    break

                remaining_rounds = max_rounds - round_number + 1
                fair_round_budget = (
                    remaining_query_budget + remaining_rounds - 1
                ) // remaining_rounds

                requested_round_queries = max(
                    1,
                    min(
                        remaining_query_budget,
                        fair_round_budget,
                        len(target_subquestions) * queries_per_subquestion,
                    ),
                )

                combined_analysis_vector = "\n".join(
                    f"- {subquestion}" for subquestion in target_subquestions
                )

                round_query_plan = self.planner.generate_search_queries(
                    target_profile=target_profile,
                    subquestion=combined_analysis_vector,
                    limit=requested_round_queries,
                    previous_queries=sorted(completed_queries),
                    round_number=round_number,
                )

                query_pool = [
                    query
                    for query in dict.fromkeys(round_query_plan.queries)
                    if query.casefold() not in completed_queries
                ][:remaining_query_budget]

                query_plans.append(
                    {
                        "round": round_number,
                        "subquestions": target_subquestions[:],
                        "queries": query_pool,
                    }
                )

                self._emit(
                    "round_started",
                    {
                        "round": round_number,
                        "max_rounds": max_rounds,
                        "planned_queries": len(query_pool),
                        "sources_collected": len(all_sources),
                    },
                )

                round_new_sources = 0
                round_queries = 0
                source_budget_exhausted = False

                for query_index, query in enumerate(query_pool):
                    if len(completed_queries) >= max_queries:
                        break

                    subquestion = target_subquestions[
                        query_index % len(target_subquestions)
                    ]

                    query_key = query.casefold()
                    completed_queries.add(query_key)
                    round_queries += 1

                    self._emit(
                        "query_started",
                        {
                            "round": round_number,
                            "query": query,
                            "subquestion": subquestion,
                        },
                    )

                    try:
                        search_results = self.search_agent.search(
                            query,
                            self.config.max_sources_per_query,
                        )
                    except Exception as exc:
                        logger.warning("Search failed '%s': %s", query, exc)
                        self._emit(
                            "query_failed",
                            {
                                "round": round_number,
                                "query": query,
                                "error": str(exc),
                            },
                        )
                        continue

                    for item in search_results:
                        url = compact_spaces(item.get("url", ""))

                        if not url:
                            continue

                        url_key = url.casefold()

                        if url_key in seen_urls:
                            continue

                        domain = domain_from_url(url)

                        if domain_counts[domain] >= DEFAULT_DOMAIN_LIMIT:
                            continue

                        evidence = self.reader.build_evidence(
                            item,
                            query=query,
                            subquestion=subquestion,
                            fetch_pages=request.fetch_pages,
                        )

                        seen_urls.add(url_key)
                        domain_counts[domain] += 1
                        all_sources.append(evidence)
                        round_new_sources += 1

                        self._emit(
                            "source_collected",
                            {
                                "round": round_number,
                                "url": url,
                                "domain": domain,
                                "sources_collected": len(all_sources),
                            },
                        )

                        if len(all_sources) >= max_sources:
                            source_budget_exhausted = True
                            break

                    if source_budget_exhausted:
                        break

                current_sources = all_sources[:max_sources]
                gap_report = self.gap_analyzer.analyze(plan, current_sources)

                round_summary = {
                    "round": round_number,
                    "queries": round_queries,
                    "new_sources": round_new_sources,
                    "sources_collected": len(current_sources),
                    "query_budget_remaining": max(
                        0,
                        max_queries - len(completed_queries),
                    ),
                    "gaps_remaining": len(gap_report.gaps),
                }

                round_summaries.append(round_summary)
                self._emit("round_completed", round_summary)

                if source_budget_exhausted:
                    break

                if not query_pool:
                    break

                target_subquestions = [
                    gap["subquestion"] for gap in gap_report.gaps
                ] or plan.subquestions[:]

            inventory_sources = all_sources[:max_sources]
            gap_report = self.gap_analyzer.analyze(plan, inventory_sources)
            duration_seconds = round(time.time() - started, 3)

            result = ResearchResult(
                topic=topic,
                target_profile=asdict(target_profile),
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                duration_seconds=duration_seconds,
                model=self.llm.model_identifier,
                engine=self.engine,
                settings={
                    "max_rounds": max_rounds,
                    "max_sources": max_sources,
                    "max_sources_per_query": self.config.max_sources_per_query,
                    "max_queries": max_queries,
                    "queries_per_subquestion": queries_per_subquestion,
                    "queries_used": len(completed_queries),
                    "max_chars_per_source": self.config.max_chars_per_source,
                    "fetch_pages": request.fetch_pages,
                    "cache_path": str(CACHE_PATH) if self.config.use_cache else None,
                },
                plan=asdict(plan),
                query_plans=query_plans,
                rounds=round_summaries,
                gap_report=asdict(gap_report),
                sources=[asdict(source) for source in inventory_sources],
            )

            self._emit(
                "completed",
                {
                    "topic": topic,
                    "duration_seconds": duration_seconds,
                    "sources": len(inventory_sources),
                    "queries_used": len(completed_queries),
                },
            )

            return result

        except Exception as exc:
            self._emit("failed", {"topic": topic, "error": str(exc)})
            raise

        finally:
            self.close()


def run_research_pipeline(
    request: ResearchRequest,
    progress_callback: ProgressCallback | None = None,
    config: Settings | None = None,
) -> ResearchResult:
    pipeline = DeepResearchPipeline(
        config=config,
        engine=request.engine,
        debug=request.debug,
        progress_callback=progress_callback,
    )
    return pipeline.run(request)


__all__ = ["DeepResearchPipeline", "ProgressCallback", "run_research_pipeline"]
