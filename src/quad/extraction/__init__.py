"""Extraction wrappers for the quad package."""

from quad.extraction.base import (
    HeuristicExtractionBackend,
    OllamaExtractionBackend,
    Reader,
    SimpleLogger,
)
from quad.extraction.config import (
    ChunkingConfig,
    ConsolidationConfig,
    ExtractionConfig,
    FieldConfig,
    ModelConfig,
    SectionConfig,
    load_extraction_config,
)
from quad.extraction.schema import Doc, DocumentExtraction, EvidenceSpan

__all__ = [
    "ChunkingConfig",
    "ConsolidationConfig",
    "Doc",
    "DocumentExtraction",
    "EvidenceSpan",
    "ExtractionConfig",
    "FieldConfig",
    "HeuristicExtractionBackend",
    "ModelConfig",
    "OllamaExtractionBackend",
    "Reader",
    "SectionConfig",
    "SimpleLogger",
    "load_extraction_config",
]
