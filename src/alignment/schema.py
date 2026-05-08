"""Structured output shapes for alignment."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any


@dataclass(slots=True)
class ChunkTrace:
    """Traceable evidence carried forward from extraction spans."""

    document_id: str
    section: str
    field_name: str
    text: str
    chunk_id: str | None = None
    page: int | None = None
    score: float = 0.0


@dataclass(slots=True)
class AlignmentInsight:
    """One traceable alignment insight."""

    text: str
    research_evidence: list[ChunkTrace] = field(default_factory=list)
    policy_evidence: list[ChunkTrace] = field(default_factory=list)


@dataclass(slots=True)
class FieldAlignment:
    """One section-field alignment judgment."""

    section: str
    field_name: str
    research_document_id: str
    policy_document_ids: list[str] = field(default_factory=list)
    shared_features: list[AlignmentInsight] = field(default_factory=list)
    policy_requirements_not_covered: list[AlignmentInsight] = field(default_factory=list)
    research_capabilities_not_used: list[AlignmentInsight] = field(default_factory=list)
    bridge_actions: list[AlignmentInsight] = field(default_factory=list)
    rationale: str = ""
    subrubric_scores: dict[str, int] = field(default_factory=dict)
    research_inputs: list[ChunkTrace] = field(default_factory=list)
    policy_inputs: list[ChunkTrace] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AlignmentReport:
    """Final alignment report for one research document against many policies."""

    schema_version: str
    research_document_id: str
    policy_document_ids: list[str] = field(default_factory=list)
    field_results: list[FieldAlignment] = field(default_factory=list)
    shared_features: list[AlignmentInsight] = field(default_factory=list)
    policy_requirements_not_covered: list[AlignmentInsight] = field(default_factory=list)
    research_capabilities_not_used: list[AlignmentInsight] = field(default_factory=list)
    bridge_actions: list[AlignmentInsight] = field(default_factory=list)
    rationale: str = ""
    subrubric_scores: dict[str, int] = field(default_factory=dict)
    dimension_scores: dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    overall_percent: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        output: dict[str, Any] = {}
        for item in fields(value):
            output[item.name] = _serialize(getattr(value, item.name))
        return output
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value
