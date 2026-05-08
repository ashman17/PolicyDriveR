"""Prompt template loading for extraction."""

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
    """One prompt definition."""

    system: str
    user: str
    schema: dict[str, Any]


@dataclass(slots=True)
class PromptLibrary:
    """All extraction prompt templates."""

    pass1: PromptSpec
    pass2: PromptSpec

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptLibrary":
        return cls(
            pass1=_prompt_spec(data["pass1"]),
            pass2=_prompt_spec(data["pass2"]),
        )


def load_prompt_library(source: str | Path | None = None) -> PromptLibrary:
    yaml_module = _get_yaml_module()
    if source is None:
        path = files("extraction").joinpath("templates/prompts.yaml")
        data = yaml_module.safe_load(path.read_text(encoding="utf-8")) or {}
        return PromptLibrary.from_dict(data)

    path = Path(source)
    data = yaml_module.safe_load(path.read_text(encoding="utf-8")) or {}
    return PromptLibrary.from_dict(data)


def _prompt_spec(data: dict[str, Any]) -> PromptSpec:
    return PromptSpec(
        system=str(data.get("system", "")),
        user=str(data.get("user", "")),
        schema=dict(data.get("schema", {})),
    )


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required for YAML-backed prompt templates. Reinstall with `pip install -e .`.")


def _get_yaml_module() -> Any:
    _require_yaml()
    assert yaml is not None
    return yaml
