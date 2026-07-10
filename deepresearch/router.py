from __future__ import annotations

from typing import Any

from .models import ResearchRequest, ResearchResult
from .service import DeepResearchService


class DeepResearchRouter:
    def __init__(self) -> None:
        self.service = DeepResearchService()

    def execute(self, request: ResearchRequest) -> ResearchResult:
        return self.service.run(request)

    def execute_from_dict(self, payload: dict[str, Any]) -> ResearchResult:
        request = ResearchRequest(**payload)
        return self.execute(request)


router = DeepResearchRouter()


def route_request(request: ResearchRequest) -> ResearchResult:
    return router.execute(request)


def route_from_dict(payload: dict[str, Any]) -> ResearchResult:
    return router.execute_from_dict(payload)


__all__ = [
    "DeepResearchRouter",
    "route_from_dict",
    "route_request",
    "router",
]
