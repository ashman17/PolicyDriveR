"""Structured output shapes for scoring."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any


@dataclass(slots=True)
class SubrubricScore:
    """One normalized subrubric score."""

    name: str
    score: int
    raw_rate: float
    normalized_rate: float
    applicable_policy_count: int
    applicable_field_count: int
    policy_document_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RubricScore:
    """One rubric-level score."""

    name: str
    score: int
    normalized_rate: float
    weight: float
    subrubric_scores: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class ScoreReport:
    """Final score report for one research document across many policy comparisons."""

    schema_version: str
    research_document_id: str
    policy_document_ids: list[str] = field(default_factory=list)
    subrubric_scores: dict[str, int] = field(default_factory=dict)
    rubric_scores: dict[str, int] = field(default_factory=dict)
    overall_score: int = 0
    overall_rate: float = 0.0
    subrubric_details: list[SubrubricScore] = field(default_factory=list)
    rubric_details: list[RubricScore] = field(default_factory=list)
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
