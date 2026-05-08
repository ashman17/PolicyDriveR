"""Configuration helpers for the extraction pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any
import yaml


@dataclass(slots=True)
class FieldConfig:
    """Config for a single extracted field."""

    name: str
    description: str
    keywords: list[str] = field(default_factory=list)
    controlled_vocab: list[str] = field(default_factory=list)
    required: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FieldConfig":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            keywords=list(data.get("keywords", [])),
            controlled_vocab=list(data.get("controlled_vocab", [])),
            required=bool(data.get("required", False)),
        )


@dataclass(slots=True)
class SectionConfig:
    """Config for a routed section."""

    name: str
    description: str
    keywords: list[str] = field(default_factory=list)
    fields: list[FieldConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SectionConfig":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            keywords=list(data.get("keywords", [])),
            fields=[FieldConfig.from_dict(item) for item in data.get("fields", [])],
        )


@dataclass(slots=True)
class ChunkingConfig:
    """Config for chunk sizing."""

    target_tokens: int = 800
    max_tokens: int = 1000
    overlap_ratio: float = 0.12
    detect_document_sections: bool = True
    excluded_section_patterns: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkingConfig":
        return cls(
            target_tokens=int(data.get("target_tokens", 800)),
            max_tokens=int(data.get("max_tokens", 1000)),
            overlap_ratio=float(data.get("overlap_ratio", 0.12)),
            detect_document_sections=bool(data.get("detect_document_sections", True)),
            excluded_section_patterns=list(
                data.get(
                    "excluded_section_patterns",
                    [
                        "related work",
                        "prior work",
                        "literature review",
                        "references",
                        "bibliography",
                        "acknowledg",
                        "appendix",
                        "supplementary",
                    ],
                )
            ),
        )


@dataclass(slots=True)
class ModelConfig:
    """Config for the local extraction backend."""

    backend: str = "ollama"
    model: str = "llama3:8b"
    comparison_model: str | None = None
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 90
    temperature: float = 0.0
    checkpoint_dir: str = "checkpoints/extraction"
    classification_max_chars: int = 3500
    include_schema_in_prompt: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(
            backend=data.get("backend", "ollama"),
            model=data.get("model", "llama3:8b"),
            comparison_model=data.get("comparison_model"),
            base_url=data.get("base_url", "http://localhost:11434"),
            timeout_seconds=int(data.get("timeout_seconds", 90)),
            temperature=float(data.get("temperature", 0.0)),
            checkpoint_dir=data.get(
                "checkpoint_dir",
                data.get("debug_dir", "checkpoints/extraction"),
            ),
            classification_max_chars=int(data.get("classification_max_chars", 3500)),
            include_schema_in_prompt=bool(data.get("include_schema_in_prompt", False)),
        )


@dataclass(slots=True)
class ConsolidationConfig:
    """Config for document-level consolidation."""

    consensus_rounds: int = 1
    max_evidence_spans_per_field: int = 3

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConsolidationConfig":
        return cls(
            consensus_rounds=max(1, int(data.get("consensus_rounds", 1))),
            max_evidence_spans_per_field=max(
                1,
                int(data.get("max_evidence_spans_per_field", 3)),
            ),
        )


@dataclass(slots=True)
class ExtractionConfig:
    """Top-level extraction pipeline config."""

    schema_version: str = "1.0"
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    prompts_path: str | None = None
    sections: list[SectionConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractionConfig":
        config = cls(
            schema_version=data.get("schema_version", "1.0"),
            chunking=ChunkingConfig.from_dict(data.get("chunking", {})),
            model=ModelConfig.from_dict(data.get("model", {})),
            consolidation=ConsolidationConfig.from_dict(data.get("consolidation", {})),
            prompts_path=data.get("prompts_path"),
            sections=[SectionConfig.from_dict(item) for item in data.get("sections", [])],
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.sections:
            raise ValueError("Extraction config must define at least one section.")
        seen_fields: set[str] = set()
        for section in self.sections:
            if not section.fields:
                raise ValueError(f"Section '{section.name}' must define at least one field.")
            for field_config in section.fields:
                if field_config.name in seen_fields:
                    raise ValueError(f"Duplicate field name '{field_config.name}' in config.")
                seen_fields.add(field_config.name)

    @property
    def section_map(self) -> dict[str, SectionConfig]:
        return {section.name: section for section in self.sections}

    @property
    def field_names(self) -> list[str]:
        return [field_config.name for section in self.sections for field_config in section.fields]

    @property
    def field_to_section(self) -> dict[str, str]:
        return {
            field_config.name: section.name
            for section in self.sections
            for field_config in section.fields
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_extraction_config(
    source: ExtractionConfig | dict[str, Any] | str | Path | None = None,
) -> ExtractionConfig:
    """Load extraction config from defaults, a mapping, or a JSON/YAML file."""

    if isinstance(source, ExtractionConfig):
        source.validate()
        return source

    base = load_default_extraction_config_dict()
    if source is None:
        return ExtractionConfig.from_dict(base)
    if isinstance(source, dict):
        return ExtractionConfig.from_dict(_deep_merge(base, source))

    path = Path(source)
    raw = _load_structured_file(path)
    return ExtractionConfig.from_dict(_deep_merge(base, raw))


def load_default_extraction_config_dict() -> dict[str, Any]:
    yaml_module = _get_yaml_module()
    default_path = files("extraction").joinpath("templates/default_extraction_config.yaml")
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
        raise RuntimeError("PyYAML is required for YAML-backed extraction config. Reinstall with `pip install -e .`.")


def _get_yaml_module() -> Any:
    _require_yaml()
    assert yaml is not None
    return yaml
