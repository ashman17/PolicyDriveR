"""Configuration helpers for the viewer pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class InputConfig:
    """Input source configuration."""

    alignment_checkpoint_dir: str = "checkpoints/alignment/final"
    scoring_checkpoint_dir: str = "checkpoints/scoring/final"
    comparison_glob: str = "*.json"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InputConfig":
        return cls(
            alignment_checkpoint_dir=data.get("alignment_checkpoint_dir", "checkpoints/alignment/final"),
            scoring_checkpoint_dir=data.get("scoring_checkpoint_dir", "checkpoints/scoring/final"),
            comparison_glob=data.get("comparison_glob", "*.json"),
        )


@dataclass(slots=True)
class OutputConfig:
    """Output persistence settings."""

    dashboard_dir: str = "checkpoints/viewer"
    document_root_dir: str = "data"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputConfig":
        return cls(
            dashboard_dir=data.get("dashboard_dir", "checkpoints/viewer"),
            document_root_dir=data.get("document_root_dir", "data"),
        )


@dataclass(slots=True)
class UiConfig:
    """Small UI-tuning settings."""

    title: str = "PolicyDriveR Dashboard"
    max_highlights_per_category: int = 6
    max_policy_insights_per_category: int = 6

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UiConfig":
        return cls(
            title=data.get("title", "PolicyDriveR Dashboard"),
            max_highlights_per_category=int(data.get("max_highlights_per_category", 6)),
            max_policy_insights_per_category=int(data.get("max_policy_insights_per_category", 6)),
        )


@dataclass(slots=True)
class ViewerConfig:
    """Top-level viewer config."""

    schema_version: str = "1.0"
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    alignment_config_path: str | None = None
    scoring_config_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ViewerConfig":
        return cls(
            schema_version=data.get("schema_version", "1.0"),
            input=InputConfig.from_dict(data.get("input", {})),
            output=OutputConfig.from_dict(data.get("output", {})),
            ui=UiConfig.from_dict(data.get("ui", {})),
            alignment_config_path=data.get("alignment_config_path"),
            scoring_config_path=data.get("scoring_config_path"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_viewer_config(
    source: ViewerConfig | dict[str, Any] | str | Path | None = None,
) -> ViewerConfig:
    """Load viewer config from defaults, a mapping, or a JSON/YAML file."""

    if isinstance(source, ViewerConfig):
        return source

    base = load_default_viewer_config_dict()
    if source is None:
        return ViewerConfig.from_dict(base)
    if isinstance(source, dict):
        return ViewerConfig.from_dict(_deep_merge(base, source))

    path = Path(source)
    raw = _load_structured_file(path)
    return ViewerConfig.from_dict(_deep_merge(base, raw))


def load_default_viewer_config_dict() -> dict[str, Any]:
    yaml_module = _get_yaml_module()
    default_path = files("viewer").joinpath("templates/default_viewer_config.yaml")
    return yaml_module.safe_load(default_path.read_text(encoding="utf-8")) or {}


def _load_structured_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        yaml_module = _get_yaml_module()
        return yaml_module.safe_load(text) or {}
    raise ValueError(f"Unsupported config format: '{path.suffix}'. Use .json or .yaml.")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = _deep_merge(merged[key], value)
            continue
        merged[key] = value
    return merged


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required for YAML-backed viewer config. Reinstall with `pip install -e .`.")


def _get_yaml_module() -> Any:
    _require_yaml()
    assert yaml is not None
    return yaml
