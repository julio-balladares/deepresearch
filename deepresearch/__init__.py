from .models import ResearchRequest, ResearchResult
from .router import (
    DeepResearchRouter,
    route_from_dict,
    route_request,
)
from .service import DeepResearchService, run_deep_research

__version__ = "1.0.0"

__all__ = [
    "DeepResearchRouter",
    "DeepResearchService",
    "ResearchRequest",
    "ResearchResult",
    "route_from_dict",
    "route_request",
    "run_deep_research",
]
