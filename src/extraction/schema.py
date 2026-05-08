"""Shared data shapes for extraction."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any


@dataclass(slots=True)
class PageText:
    """Text content for one page."""

    page_number: int
    text: str


@dataclass(slots=True)
class RawDocument:
    """Raw document after file loading and text extraction."""

    document_id: str
    title: str
    source_path: str | None = None
    pages: list[PageText] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)


@dataclass(slots=True)
class Chunk:
    """Chunked document unit for multi-pass extraction."""

    chunk_id: str
    text: str
    token_count: int
    page_start: int | None = None
    page_end: int | None = None
    source_pages: list[int] = field(default_factory=list)


@dataclass(slots=True)
class ChunkClassification:
    """Pass-1 routing result."""

    chunk_id: str
    labels: list[str] = field(default_factory=list)
    confidence: float = 0.0
    label_scores: dict[str, float] = field(default_factory=dict)
    page_start: int | None = None
    page_end: int | None = None


@dataclass(slots=True)
class EvidenceSpan:
    """Short evidence anchor for a field value."""

    field_name: str
    text: str
    section: str
    page: int | None = None
    chunk_id: str | None = None
    score: float = 0.0


@dataclass(slots=True)
class ChunkExtraction:
    """Pass-2 extraction result for a routed chunk/section pair."""

    chunk_id: str
    section: str
    fields: dict[str, str] = field(default_factory=dict)
    evidence_spans: list[EvidenceSpan] = field(default_factory=list)
    extraction_notes: str = ""
    confidence: float = 0.0
    round_index: int = 0


@dataclass(slots=True)
class FieldCandidate:
    """Intermediate candidate used during consolidation."""

    field_name: str
    value: str
    section: str
    source_chunk_ids: list[str] = field(default_factory=list)
    evidence_spans: list[EvidenceSpan] = field(default_factory=list)
    vote_weight: float = 0.0
    round_index: int = 0


@dataclass(slots=True)
class Doc:
    """Final structured extraction output."""

    schema_version: str
    document_id: str
    title: str
    source_path: str | None = None
    sections: dict[str, dict[str, str]] = field(default_factory=dict)
    fields: dict[str, str] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    evidence_spans: list[EvidenceSpan] = field(default_factory=list)
    chunk_classification: list[ChunkClassification] = field(default_factory=list)
    extraction_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def docid(self) -> str:
        return self.document_id

    @property
    def kind(self) -> str:
        return "extraction"

    @property
    def meta(self) -> dict[str, Any]:
        return self.metadata

    @property
    def body(self) -> dict[str, Any]:
        return self.sections

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


DocumentExtraction = Doc


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
