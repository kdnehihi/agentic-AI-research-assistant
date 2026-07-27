from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExecutionStrategy(str, Enum):
    """Supervisor-level route for one user request."""

    KNOWLEDGE_ONLY = "knowledge_only"
    DISCOVERY_ONLY = "discovery_only"
    DISCOVER_THEN_ANSWER = "discover_then_answer"


KnowledgeCoverage = Literal["sufficient", "partial", "insufficient"]


class KnowledgeCoverageDecision(BaseModel):
    """Deterministic judgment of whether indexed KB evidence can answer."""

    model_config = ConfigDict(extra="forbid")

    coverage: KnowledgeCoverage
    retrieved_evidence_ids: list[str] = Field(default_factory=list)
    relevant_paper_ids: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    recommended_strategy: ExecutionStrategy
