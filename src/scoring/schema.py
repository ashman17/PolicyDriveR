"""Scoring placeholders."""

from dataclasses import dataclass, field


@dataclass
class Card:
    """Minimal scorecard."""

    total: int
    band: str
    tips: list[str] = field(default_factory=list)
