"""Prompt template loading for alignment."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - depends on installed dependencies
    yaml = None


@dataclass(slots=True)
class PromptSpec:
    """One alignment prompt definition."""

    system: str
    user: str


@dataclass(slots=True)
class PromptLibrary:
    """All alignment prompt templates."""

    comparison: PromptSpec

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptLibrary":
        return cls(comparison=_prompt_spec(data["comparison"]))


def load_prompt_library(source: str | Path | None = None) -> PromptLibrary:
    yaml_module = _get_yaml_module()
    if source is None:
        path = files("alignment").joinpath("templates/prompts.yaml")
        data = yaml_module.safe_load(path.read_text(encoding="utf-8")) or {}
        return PromptLibrary.from_dict(data)

    path = Path(source)
    data = yaml_module.safe_load(path.read_text(encoding="utf-8")) or {}
    return PromptLibrary.from_dict(data)


def _prompt_spec(data: dict[str, Any]) -> PromptSpec:
    return PromptSpec(
        system=str(data.get("system", "")),
        user=str(data.get("user", "")),
    )


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for YAML-backed alignment prompt templates. Reinstall with `pip install -e .`."
        )


def _get_yaml_module() -> Any:
    _require_yaml()
    assert yaml is not None
    return yaml
