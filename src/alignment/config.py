"""Configuration helpers for the alignment pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class OutputFieldConfig:
    """One required structured output field."""

    name: str
    description: str
    kind: str = "list"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputFieldConfig":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            kind=data.get("kind", "list"),
        )


@dataclass(slots=True)
class SubrubricConfig:
    """One binary-scored alignment subrubric."""

    name: str
    description: str
    weight: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubrubricConfig":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            weight=float(data.get("weight", 0.0)),
        )


@dataclass(slots=True)
class RubricConfig:
    """One weighted alignment rubric dimension."""

    name: str
    description: str
    weight: float = 0.0
    subrubrics: list[SubrubricConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RubricConfig":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            weight=float(data.get("weight", 0.0)),
            subrubrics=[SubrubricConfig.from_dict(item) for item in data.get("subrubrics", [])],
        )


@dataclass(slots=True)
class SectionFieldTargetConfig:
    """Field-to-rubric focus mapping for prompt construction."""

    section: str
    fields: list[str] = field(default_factory=list)
    focus_subrubrics: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SectionFieldTargetConfig":
        return cls(
            section=data["section"],
            fields=list(data.get("fields", [])),
            focus_subrubrics=list(data.get("focus_subrubrics", [])),
        )


@dataclass(slots=True)
class InputConfig:
    """Input source configuration."""

    extraction_checkpoint_dir: str = "checkpoints/extraction/pass3"
    policy_folder_glob: str = "policy*"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InputConfig":
        return cls(
            extraction_checkpoint_dir=data.get(
                "extraction_checkpoint_dir",
                "checkpoints/extraction/pass3",
            ),
            policy_folder_glob=data.get("policy_folder_glob", "policy*"),
        )


@dataclass(slots=True)
class ModelConfig:
    """Local LLM backend settings for alignment."""

    backend: str = "ollama"
    model: str = "llama3:8b"
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 90
    temperature: float = 0.0
    checkpoint_dir: str = "checkpoints/alignment"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(
            backend=data.get("backend", "ollama"),
            model=data.get("model", "llama3:8b"),
            base_url=data.get("base_url", "http://localhost:11434"),
            timeout_seconds=int(data.get("timeout_seconds", 90)),
            temperature=float(data.get("temperature", 0.0)),
            checkpoint_dir=data.get("checkpoint_dir", "checkpoints/alignment"),
        )


@dataclass(slots=True)
class AlignmentConfig:
    """Top-level alignment config."""

    schema_version: str = "1.0"
    input: InputConfig = field(default_factory=InputConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    prompts_path: str | None = None
    output_fields: list[OutputFieldConfig] = field(default_factory=list)
    rubrics: list[RubricConfig] = field(default_factory=list)
    section_field_targets: list[SectionFieldTargetConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlignmentConfig":
        config = cls(
            schema_version=data.get("schema_version", "1.0"),
            input=InputConfig.from_dict(data.get("input", {})),
            model=ModelConfig.from_dict(data.get("model", {})),
            prompts_path=data.get("prompts_path"),
            output_fields=[OutputFieldConfig.from_dict(item) for item in data.get("output_fields", [])],
            rubrics=[RubricConfig.from_dict(item) for item in data.get("rubrics", [])],
            section_field_targets=[
                SectionFieldTargetConfig.from_dict(item)
                for item in data.get("section_field_targets", [])
            ],
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.output_fields:
            raise ValueError("Alignment config must define at least one output field.")
        if not self.rubrics:
            raise ValueError("Alignment config must define at least one rubric.")
        if not self.section_field_targets:
            raise ValueError("Alignment config must define at least one section-field target.")
        seen_subrubrics: set[str] = set()
        for rubric in self.rubrics:
            if not rubric.subrubrics:
                raise ValueError(f"Rubric '{rubric.name}' must define at least one subrubric.")
            for subrubric in rubric.subrubrics:
                if subrubric.name in seen_subrubrics:
                    raise ValueError(f"Duplicate subrubric '{subrubric.name}' in alignment config.")
                seen_subrubrics.add(subrubric.name)

    @property
    def all_subrubrics(self) -> list[SubrubricConfig]:
        return [subrubric for rubric in self.rubrics for subrubric in rubric.subrubrics]

    @property
    def subrubric_names(self) -> list[str]:
        return [subrubric.name for subrubric in self.all_subrubrics]

    @property
    def subrubric_map(self) -> dict[str, SubrubricConfig]:
        return {subrubric.name: subrubric for subrubric in self.all_subrubrics}

    @property
    def rubric_by_subrubric(self) -> dict[str, RubricConfig]:
        return {
            subrubric.name: rubric
            for rubric in self.rubrics
            for subrubric in rubric.subrubrics
        }

    @property
    def target_map(self) -> dict[tuple[str, str], list[str]]:
        mapping: dict[tuple[str, str], list[str]] = {}
        for target in self.section_field_targets:
            for field_name in target.fields:
                mapping[(target.section, field_name)] = list(target.focus_subrubrics)
        return mapping

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_alignment_config(
    source: AlignmentConfig | dict[str, Any] | str | Path | None = None,
) -> AlignmentConfig:
    """Load alignment config from defaults, a mapping, or a JSON/YAML file."""

    if isinstance(source, AlignmentConfig):
        source.validate()
        return source

    base = load_default_alignment_config_dict()
    if source is None:
        return AlignmentConfig.from_dict(base)
    if isinstance(source, dict):
        return AlignmentConfig.from_dict(_deep_merge(base, source))

    path = Path(source)
    raw = _load_structured_file(path)
    return AlignmentConfig.from_dict(_deep_merge(base, raw))


def load_default_alignment_config_dict() -> dict[str, Any]:
    yaml_module = _get_yaml_module()
    default_path = files("alignment").joinpath("templates/default_alignment_config.yaml")
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
        if (
            isinstance(value, dict)
            and key in merged
            and isinstance(merged[key], dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
            continue
        merged[key] = value
    return merged


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for YAML-backed alignment config. Reinstall with `pip install -e .`."
        )


def _get_yaml_module() -> Any:
    _require_yaml()
    assert yaml is not None
    return yaml
