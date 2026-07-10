from __future__ import annotations

from typing import Any, Callable

from .config import Settings, settings
from .models import ResearchRequest, ResearchResult
from .pipeline import DeepResearchPipeline

ProgressCallback = Callable[[str, dict[str, Any]], None]


class DeepResearchService:
    def __init__(
        self,
        config: Settings | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.config = config or settings
        self.progress_callback = progress_callback

    def run(self, request: ResearchRequest) -> ResearchResult:
        pipeline = DeepResearchPipeline(
            config=self.config,
            engine=request.engine,
            debug=request.debug,
            progress_callback=self.progress_callback,
        )

        return pipeline.run(request)


def run_deep_research(
    request: ResearchRequest,
    progress_callback: ProgressCallback | None = None,
    config: Settings | None = None,
) -> ResearchResult:
    service = DeepResearchService(
        config=config,
        progress_callback=progress_callback,
    )

    return service.run(request)


__all__ = [
    "DeepResearchService",
    "ProgressCallback",
    "run_deep_research",
]
