"""Configuration helpers for the scoring pipeline."""

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
    comparison_glob: str = "*.json"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InputConfig":
        return cls(
            alignment_checkpoint_dir=data.get("alignment_checkpoint_dir", "checkpoints/alignment/final"),
            comparison_glob=data.get("comparison_glob", "*.json"),
        )


@dataclass(slots=True)
class NormalizationConfig:
    """Normalization settings for multi-policy score aggregation."""

    prior_rate: float = 0.5
    prior_strength: float = 2.0
    pair_weight_mode: str = "applicable_field_count"
    focus_metadata_key: str = "focus_subrubrics"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NormalizationConfig":
        return cls(
            prior_rate=float(data.get("prior_rate", 0.5)),
            prior_strength=float(data.get("prior_strength", 2.0)),
            pair_weight_mode=data.get("pair_weight_mode", "applicable_field_count"),
            focus_metadata_key=data.get("focus_metadata_key", "focus_subrubrics"),
        )


@dataclass(slots=True)
class OutputConfig:
    """Output persistence settings."""

    checkpoint_dir: str = "checkpoints/scoring"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputConfig":
        return cls(checkpoint_dir=data.get("checkpoint_dir", "checkpoints/scoring"))


@dataclass(slots=True)
class ScoringConfig:
    """Top-level scoring config."""

    schema_version: str = "1.0"
    input: InputConfig = field(default_factory=InputConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    alignment_config_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoringConfig":
        return cls(
            schema_version=data.get("schema_version", "1.0"),
            input=InputConfig.from_dict(data.get("input", {})),
            normalization=NormalizationConfig.from_dict(data.get("normalization", {})),
            output=OutputConfig.from_dict(data.get("output", {})),
            alignment_config_path=data.get("alignment_config_path"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_scoring_config(
    source: ScoringConfig | dict[str, Any] | str | Path | None = None,
) -> ScoringConfig:
    """Load scoring config from defaults, a mapping, or a JSON/YAML file."""

    if isinstance(source, ScoringConfig):
        return source

    base = load_default_scoring_config_dict()
    if source is None:
        return ScoringConfig.from_dict(base)
    if isinstance(source, dict):
        return ScoringConfig.from_dict(_deep_merge(base, source))

    path = Path(source)
    raw = _load_structured_file(path)
    return ScoringConfig.from_dict(_deep_merge(base, raw))


def load_default_scoring_config_dict() -> dict[str, Any]:
    yaml_module = _get_yaml_module()
    default_path = files("scoring").joinpath("templates/default_scoring_config.yaml")
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
        raise RuntimeError("PyYAML is required for YAML-backed scoring config. Reinstall with `pip install -e .`.")


def _get_yaml_module() -> Any:
    _require_yaml()
    assert yaml is not None
    return yaml
