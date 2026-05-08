"""Ontology-facing placeholder shapes."""

from dataclasses import dataclass, field


@dataclass
class Term:
    """Minimal normalized term record."""

    name: str
    tags: list[str] = field(default_factory=list)
