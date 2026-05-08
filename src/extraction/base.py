"""Structured extraction layer."""

from __future__ import annotations

import json
import re
import socket
import sys
import textwrap
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

from extraction.config import ExtractionConfig, FieldConfig, SectionConfig, load_extraction_config
from extraction.prompts import load_prompt_library
from extraction.schema import (
    Chunk,
    ChunkClassification,
    ChunkExtraction,
    Doc,
    EvidenceSpan,
    FieldCandidate,
    PageText,
    RawDocument,
)
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - depends on installed extras
    def tqdm(iterable: Any | None = None, *args: Any, total: int | None = None, **kwargs: Any) -> Any:
        del args, kwargs
        if iterable is None:
            return _NullProgress(total=total)
        return iterable


class _NullProgress:
    def __init__(self, total: int | None = None) -> None:
        self.total = total

    def __enter__(self) -> "_NullProgress":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb

    def update(self, step: int = 1) -> None:
        del step

    def set_postfix_str(self, value: str) -> None:
        del value

WORD_RE = re.compile(r"\w+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
HEADING_PREFIX_RE = re.compile(r"^(?:\d+(?:\.\d+)*|[IVXLC]+)\s+")


class SimpleLogger:
    """Small logger with short stage-based messages."""

    def __init__(self, enabled: bool = True, stream: Any | None = None) -> None:
        self.enabled = enabled
        self.stream = stream or sys.stderr

    def log(self, stage: str, message: str) -> None:
        if not self.enabled:
            return
        print(f"[extract:{stage}] {message}", file=self.stream)


class ExtractionBackend(Protocol):
    """Backend contract for routing, local extraction, and reconciliation."""

    name: str

    def classify_chunk(
        self,
        chunk: Chunk,
        sections: list[SectionConfig],
    ) -> ChunkClassification:
        ...

    def extract_fields(
        self,
        chunk: Chunk,
        section: SectionConfig,
        *,
        round_index: int = 0,
    ) -> ChunkExtraction:
        ...


class DocumentLoader:
    """Load PDFs or text files into paged text."""

    def __init__(self, logger: SimpleLogger | None = None) -> None:
        self.logger = logger or SimpleLogger(enabled=False)

    def load(self, source: str | Path, document_id: str | None = None) -> RawDocument:
        path_exists = False
        if isinstance(source, Path):
            path = source
            path_exists = path.exists()
        else:
            raw_source = str(source)
            path = Path(raw_source)
            if "\n" not in raw_source and len(raw_source) < 240:
                try:
                    path_exists = path.exists()
                except OSError:
                    path_exists = False

        if path_exists:
            if path.suffix.lower() == ".pdf":
                pages = self._load_pdf(path)
            else:
                text = path.read_text(encoding="utf-8")
                pages = self._pages_from_text(text)
            title = self._guess_title(pages, path.stem)
            docid = document_id or path.stem
            self.logger.log("read", f"loaded {len(pages)} page(s) from {path.name}")
            return RawDocument(
                document_id=docid,
                title=title,
                source_path=str(path),
                pages=pages,
            )

        text = str(source)
        pages = self._pages_from_text(text)
        title = self._guess_title(pages, "inline document")
        docid = document_id or "inline-document"
        self.logger.log("read", f"loaded inline text with {len(pages)} page(s)")
        return RawDocument(document_id=docid, title=title, pages=pages)

    def _pages_from_text(self, text: str) -> list[PageText]:
        pieces = [item.strip() for item in text.split("\f")]
        pages = [PageText(page_number=index + 1, text=piece) for index, piece in enumerate(pieces) if piece]
        if pages:
            return pages
        return [PageText(page_number=1, text=text.strip())]

    def _guess_title(self, pages: list[PageText], fallback: str) -> str:
        for page in pages:
            for line in page.text.splitlines():
                cleaned = _normalize_space(line)
                if cleaned:
                    return cleaned[:180]
        return fallback

    def _load_pdf(self, path: Path) -> list[PageText]:
        try:
            return self._load_pdf_with_pymupdf(path)
        except Exception:
            return self._load_pdf_with_unstructured(path)

    def _load_pdf_with_pymupdf(self, path: Path) -> list[PageText]:
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError("PyMuPDF is not installed.") from exc

        pages: list[PageText] = []
        with fitz.open(path) as pdf:
            for page_index in range(len(pdf)):
                page = pdf.load_page(page_index)
                text = _normalize_space(str(page.get_text("text")))
                pages.append(PageText(page_number=page_index + 1, text=text))
        if not any(page.text for page in pages):
            raise RuntimeError(f"No text extracted from PDF '{path.name}' with PyMuPDF.")
        return pages

    def _load_pdf_with_unstructured(self, path: Path) -> list[PageText]:
        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "PDF extraction requires PyMuPDF or unstructured."
            ) from exc

        per_page: dict[int, list[str]] = defaultdict(list)
        for element in partition_pdf(filename=str(path)):
            page_number = getattr(getattr(element, "metadata", None), "page_number", 1) or 1
            text = _normalize_space(str(element))
            if text:
                per_page[page_number].append(text)
        pages = [
            PageText(page_number=page_number, text="\n\n".join(items))
            for page_number, items in sorted(per_page.items())
        ]
        if not pages:
            raise RuntimeError(f"No text extracted from PDF '{path.name}' with unstructured.")
        return pages


class SemanticChunker:
    """Simple paragraph-aware chunker with configurable overlap."""

    def __init__(self, config: ExtractionConfig, logger: SimpleLogger | None = None) -> None:
        self.config = config
        self.logger = logger or SimpleLogger(enabled=False)

    def chunk(self, document: RawDocument) -> list[Chunk]:
        units = self._build_units(document)
        if not units:
            return []

        target = self.config.chunking.target_tokens
        max_tokens = self.config.chunking.max_tokens
        overlap_ratio = self.config.chunking.overlap_ratio
        chunks: list[Chunk] = []
        start = 0

        while start < len(units):
            cursor = start
            token_count = 0
            selected: list[dict[str, Any]] = []
            while cursor < len(units):
                next_tokens = token_count + units[cursor]["tokens"]
                if selected and next_tokens > max_tokens:
                    break
                selected.append(units[cursor])
                token_count = next_tokens
                cursor += 1
                if token_count >= target:
                    break

            pages = sorted({unit["page"] for unit in selected})
            chunk_text = "\n\n".join(unit["text"] for unit in selected)
            chunks.append(
                Chunk(
                    chunk_id=f"c{len(chunks) + 1:03d}",
                    text=chunk_text,
                    token_count=token_count,
                    page_start=pages[0] if pages else None,
                    page_end=pages[-1] if pages else None,
                    source_pages=pages,
                )
            )

            if cursor >= len(units):
                break

            overlap_target = max(1, int(token_count * overlap_ratio))
            overlap_count = 0
            overlap_tokens = 0
            index = cursor - 1
            while index >= start and overlap_tokens < overlap_target:
                overlap_tokens += units[index]["tokens"]
                overlap_count += 1
                index -= 1
            start = max(start + 1, cursor - overlap_count)

        self.logger.log(
            "chunk",
            f"built {len(chunks)} chunk(s) at ~{target} tokens with {int(overlap_ratio * 100)}% overlap",
        )
        return chunks

    def _build_units(self, document: RawDocument) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []
        current_section = ""
        kept_blocks = 0
        skipped_blocks = 0
        for page in document.pages:
            blocks = [item.strip() for item in re.split(r"\n\s*\n+", page.text) if item.strip()]
            for block in blocks:
                cleaned = _normalize_space(block)
                if not cleaned:
                    continue
                next_section = self._extract_section_heading(cleaned)
                if next_section:
                    current_section = next_section
                if self._should_skip_block(cleaned, current_section):
                    skipped_blocks += 1
                    continue
                kept_blocks += 1
                if _estimate_tokens(cleaned) <= self.config.chunking.max_tokens:
                    units.append(
                        {
                            "page": page.page_number,
                            "text": cleaned,
                            "tokens": _estimate_tokens(cleaned),
                        }
                    )
                    continue
                for sentence_block in self._split_long_block(cleaned, page.page_number):
                    units.append(sentence_block)
        if skipped_blocks:
            self.logger.log(
                "filter",
                f"kept {kept_blocks} block(s), skipped {skipped_blocks} block(s) in excluded sections",
            )
        return units

    def _extract_section_heading(self, text: str) -> str:
        if not self.config.chunking.detect_document_sections:
            return ""
        candidate = HEADING_PREFIX_RE.sub("", text).strip(" :.-").lower()
        if not candidate:
            return ""
        if len(candidate.split()) > 8:
            return ""
        if len(text) > 80:
            return ""
        if not any(char.isalpha() for char in candidate):
            return ""
        heading_like = text.isupper() or text.istitle() or bool(HEADING_PREFIX_RE.match(text))
        return candidate if heading_like else ""

    def _should_skip_block(self, text: str, current_section: str) -> bool:
        del text
        patterns = self.config.chunking.excluded_section_patterns
        if not self.config.chunking.detect_document_sections or not current_section or not patterns:
            return False
        return any(pattern.lower() in current_section for pattern in patterns)

    def _split_long_block(self, text: str, page_number: int) -> list[dict[str, Any]]:
        sentences = [item.strip() for item in SENTENCE_RE.split(text) if item.strip()]
        if not sentences:
            return [{"page": page_number, "text": text, "tokens": _estimate_tokens(text)}]

        output: list[dict[str, Any]] = []
        current: list[str] = []
        tokens = 0
        for sentence in sentences:
            sentence_tokens = _estimate_tokens(sentence)
            if current and tokens + sentence_tokens > self.config.chunking.max_tokens:
                joined = " ".join(current)
                output.append(
                    {"page": page_number, "text": joined, "tokens": _estimate_tokens(joined)}
                )
                current = [sentence]
                tokens = sentence_tokens
                continue
            current.append(sentence)
            tokens += sentence_tokens
        if current:
            joined = " ".join(current)
            output.append({"page": page_number, "text": joined, "tokens": _estimate_tokens(joined)})
        return output


class HeuristicExtractionBackend:
    """Cheap local backend used for tests and offline fallback."""

    name = "heuristic"

    def classify_chunk(
        self,
        chunk: Chunk,
        sections: list[SectionConfig],
    ) -> ChunkClassification:
        text = chunk.text.lower()
        label_scores: dict[str, float] = {}
        for section in sections:
            score = 0.0
            section_hits = sum(1 for keyword in section.keywords if keyword.lower() in text)
            field_hits = sum(
                1
                for field_config in section.fields
                for keyword in ([field_config.name.replace("_", " ")] + field_config.keywords)
                if keyword.lower() in text
            )
            heading_bonus = 0.35 if re.search(rf"\b{re.escape(section.name)}\b\s*:", text[:300]) else 0.0
            if section.keywords:
                score += 0.45 * min(1.0, section_hits / max(1, len(section.keywords) / 3))
            if section.fields:
                score += 0.35 * min(1.0, field_hits / max(1, len(section.fields)))
            score += heading_bonus
            if score > 0:
                label_scores[section.name] = round(min(1.0, score), 2)

        ordered = sorted(label_scores.items(), key=lambda item: item[1], reverse=True)
        labels = [name for name, score in ordered if score >= 0.2]
        if not labels and ordered:
            best_name, best_score = ordered[0]
            if best_score >= 0.1:
                labels = [best_name]
        selected_scores = {label: label_scores[label] for label in labels}
        confidence = max(selected_scores.values(), default=0.0)
        return ChunkClassification(
            chunk_id=chunk.chunk_id,
            labels=labels,
            confidence=confidence,
            label_scores=selected_scores,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
        )

    def extract_fields(
        self,
        chunk: Chunk,
        section: SectionConfig,
        *,
        round_index: int = 0,
    ) -> ChunkExtraction:
        sentences = [item.strip() for item in SENTENCE_RE.split(chunk.text) if item.strip()]
        text_lower = chunk.text.lower()
        fields: dict[str, str] = {}
        evidence: list[EvidenceSpan] = []

        for field_config in section.fields:
            value, spans = self._extract_field_value(
                chunk=chunk,
                section=section,
                field_config=field_config,
                sentences=sentences,
                text_lower=text_lower,
            )
            fields[field_config.name] = value
            evidence.extend(spans)

        note = "heuristic fallback" if any(fields.values()) else "no section-specific evidence found"
        return ChunkExtraction(
            chunk_id=chunk.chunk_id,
            section=section.name,
            fields=fields,
            evidence_spans=evidence,
            extraction_notes=note,
            round_index=round_index,
        )

    def _extract_field_value(
        self,
        *,
        chunk: Chunk,
        section: SectionConfig,
        field_config: FieldConfig,
        sentences: list[str],
        text_lower: str,
    ) -> tuple[str, list[EvidenceSpan]]:
        if field_config.controlled_vocab:
            matches = [
                term
                for term in field_config.controlled_vocab
                if term.lower() in text_lower
            ]
            if matches:
                unique_matches = _unique_preserve_order(matches)
                return ", ".join(unique_matches), [
                    EvidenceSpan(
                        field_name=field_config.name,
                        text=self._find_first_sentence(sentences, unique_matches[0]),
                        section=section.name,
                        page=chunk.page_start,
                        chunk_id=chunk.chunk_id,
                        score=0.7,
                    )
                ]

        label_variants = [
            field_config.name.replace("_", " "),
            field_config.name.replace("_", "-"),
        ]
        for variant in label_variants:
            match = re.search(
                rf"{re.escape(variant)}\s*:\s*(.+?)(?:$|\n)",
                chunk.text,
                flags=re.IGNORECASE,
            )
            if match:
                value = _normalize_space(match.group(1))
                if value:
                    return value, [
                        EvidenceSpan(
                            field_name=field_config.name,
                            text=value,
                            section=section.name,
                            page=chunk.page_start,
                            chunk_id=chunk.chunk_id,
                            score=0.8,
                        )
                    ]

        candidates = [
            sentence
            for sentence in sentences
            if any(keyword.lower() in sentence.lower() for keyword in field_config.keywords)
        ]
        if not candidates and any(keyword.lower() in text_lower for keyword in section.keywords):
            candidates = sentences[:1]
        if not candidates:
            return "", []

        selected = _normalize_space(" ".join(candidates[:2]))
        spans = [
            EvidenceSpan(
                field_name=field_config.name,
                text=candidate,
                section=section.name,
                page=chunk.page_start,
                chunk_id=chunk.chunk_id,
                score=0.6,
            )
            for candidate in candidates[:2]
        ]
        return selected, spans

    def _find_first_sentence(self, sentences: list[str], keyword: str) -> str:
        for sentence in sentences:
            if keyword.lower() in sentence.lower():
                return sentence
        return sentences[0] if sentences else keyword


class OllamaExtractionBackend:
    """Schema-guided JSON prompting against a local Ollama endpoint."""

    name = "ollama"

    def __init__(self, config: ExtractionConfig) -> None:
        self.config = config
        self.prompts = load_prompt_library(self.config.prompts_path)
        self.checkpoints = CheckpointStore(
            root_dir=Path(self.config.model.checkpoint_dir),
        )

    def prepare_document(self, document: RawDocument) -> None:
        self.checkpoints.set_scope(document.document_id, document.title)

    def classify_chunk(
        self,
        chunk: Chunk,
        sections: list[SectionConfig],
    ) -> ChunkClassification:
        schema = json.loads(json.dumps(self.prompts.pass1.schema))
        section_help = "\n".join(
            f"- {section.name}: {section.description}; fields: {', '.join(field.name for field in section.fields)}"
            for section in sections
        )
        payload = self._complete_json(
            model_name=self.config.model.model,
            schema=schema,
            pass_name="pass1",
            file_stem=f"classify_{chunk.chunk_id}_{self.config.model.model}",
            system_prompt=self.prompts.pass1.system.format(section_help=section_help),
            user_prompt=self.prompts.pass1.user.format(
                chunk_text=self._truncate_for_classification(chunk.text),
            ),
        )
        allowed = {section.name for section in sections}
        labels = _coerce_list(payload.get("labels"))
        labels = [label for label in labels if label in allowed]
        raw_scores = payload.get("label_scores", {})
        if not labels and isinstance(raw_scores, dict):
            ranked = sorted(
                (
                    (label, _clamp(score))
                    for label, score in raw_scores.items()
                    if label in allowed
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            labels = [label for label, score in ranked if score >= 0.2]
        label_scores = {
            label: _clamp(raw_scores.get(label, 0.0))
            for label in labels
        }
        confidence = _clamp(payload.get("confidence", max(label_scores.values(), default=0.0)))
        return ChunkClassification(
            chunk_id=chunk.chunk_id,
            labels=labels,
            confidence=confidence,
            label_scores=label_scores,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
        )

    def extract_fields(
        self,
        chunk: Chunk,
        section: SectionConfig,
        *,
        round_index: int = 0,
    ) -> ChunkExtraction:
        field_properties = {field_config.name: {"type": "string"} for field_config in section.fields}
        schema = json.loads(json.dumps(self.prompts.pass2.schema))
        schema["properties"]["fields"]["properties"] = field_properties
        schema["properties"]["fields"]["required"] = list(field_properties)
        field_help = "\n".join(
            self._field_prompt_line(field_config) for field_config in section.fields
        )
        model_name = self._model_for_round(round_index)
        payload = self._complete_json(
            model_name=model_name,
            schema=schema,
            pass_name="pass2",
            file_stem=f"extract_{chunk.chunk_id}_{section.name}_round{round_index + 1}_{model_name}",
            system_prompt=self.prompts.pass2.system,
            user_prompt=self.prompts.pass2.user.format(
                section_name=section.name,
                section_description=section.description,
                field_help=field_help,
                chunk_text=chunk.text,
            ),
        )
        raw_fields = payload.get("fields", {})
        fields = {
            field_config.name: _normalize_space(_coerce_text(raw_fields.get(field_config.name, "")))
            for field_config in section.fields
        }
        evidence_spans = [
            EvidenceSpan(
                field_name=_coerce_text(span.get("field_name", "")),
                text=_normalize_space(_coerce_text(span.get("text", ""))),
                section=_coerce_text(span.get("section", section.name)),
                page=int(span.get("page", chunk.page_start or 1)),
                chunk_id=chunk.chunk_id,
                score=0.7,
            )
            for span in payload.get("evidence_spans", [])
            if _coerce_text(span.get("field_name", "")) in fields
            and _normalize_space(_coerce_text(span.get("text", "")))
        ]
        return ChunkExtraction(
            chunk_id=chunk.chunk_id,
            section=section.name,
            fields=fields,
            evidence_spans=evidence_spans,
            extraction_notes=_normalize_space(_coerce_text(payload.get("extraction_notes", ""))),
            round_index=round_index,
        )

    def _field_prompt_line(self, field_config: FieldConfig) -> str:
        vocab = ""
        if field_config.controlled_vocab:
            vocab = f" Controlled vocab hints: {', '.join(field_config.controlled_vocab)}."
        return f"- {field_config.name}: {field_config.description}.{vocab}"

    def _truncate_for_classification(self, text: str) -> str:
        max_chars = max(500, self.config.model.classification_max_chars)
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "\n...[truncated for routing]"

    def _model_for_round(self, round_index: int) -> str:
        if round_index == 0 or not self.config.model.comparison_model:
            return self.config.model.model
        return self.config.model.comparison_model

    def _complete_json(
        self,
        *,
        model_name: str,
        schema: dict[str, Any],
        pass_name: str,
        file_stem: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        url = f"{self.config.model.base_url.rstrip('/')}/api/chat"
        rendered_user_prompt = user_prompt
        if self.config.model.include_schema_in_prompt:
            rendered_user_prompt = (
                f"{user_prompt}\n\n"
                "Output schema:\n"
                f"{json.dumps(schema, indent=2)}"
            )
        body = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": rendered_user_prompt,
                },
            ],
            "stream": False,
            "format": schema,
            "options": {"temperature": self.config.model.temperature},
        }
        cached = self.checkpoints.load_cached_parsed(pass_name=pass_name, file_stem=file_stem)
        if cached is not None:
            return cached
        checkpoint_prefix = self.checkpoints.write_request(
            pass_name=pass_name,
            file_stem=file_stem,
            model_name=model_name,
            request_url=url,
            system_prompt=system_prompt,
            user_prompt=rendered_user_prompt,
            request_body=body,
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.model.timeout_seconds,
            ) as response:
                raw_response_text = response.read().decode("utf-8")
                payload = json.loads(raw_response_text)
        except urllib.error.HTTPError as exc:  # pragma: no cover - depends on runtime service
            error_body = exc.read().decode("utf-8", errors="replace")
            self.checkpoints.write_error(
                checkpoint_prefix,
                message=f"HTTPError {exc.code}: {exc.reason}\n\n{error_body}",
            )
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code} during {pass_name}. "
                f"See checkpoint files under {self.config.model.checkpoint_dir}/{pass_name}."
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            self.checkpoints.write_error(
                checkpoint_prefix,
                message=(
                    f"Timeout after {self.config.model.timeout_seconds}s during {pass_name} "
                    f"for model {model_name}: {exc}"
                ),
            )
            raise RuntimeError(
                f"Ollama request timed out after {self.config.model.timeout_seconds}s during {pass_name}. "
                "Try a smaller model, a larger --timeout-seconds, or smaller chunks."
            ) from exc
        except urllib.error.URLError as exc:  # pragma: no cover - depends on runtime service
            self.checkpoints.write_error(
                checkpoint_prefix,
                message=f"{type(exc).__name__}: {exc}",
            )
            raise RuntimeError(
                "Could not reach Ollama. Start the Ollama service locally or point base_url to the correct hosted endpoint."
            ) from exc
        self.checkpoints.write_response(
            checkpoint_prefix,
            response_body=payload,
            raw_response_text=raw_response_text,
        )
        content = payload.get("message", {}).get("content", "")
        parsed = _extract_json(content)
        self.checkpoints.write_parsed(checkpoint_prefix, parsed)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Model returned non-object JSON for schema {schema!r}")
        return parsed


class Reader:
    """Configurable multi-pass extraction pipeline."""

    def __init__(
        self,
        config: ExtractionConfig | dict[str, Any] | str | Path | None = None,
        backend: ExtractionBackend | None = None,
        logger: SimpleLogger | None = None,
    ) -> None:
        self.config = load_extraction_config(config)
        self.logger = logger or SimpleLogger()
        self.loader = DocumentLoader(logger=self.logger)
        self.chunker = SemanticChunker(self.config, logger=self.logger)
        self.backend = backend or self._build_backend()
        self.logger.log(
            "model",
            f"backend={self.backend.name} primary={self.config.model.model} comparison={self.config.model.comparison_model}",
        )
        self.logger.log("checkpoint", f"writing llm checkpoints to {self.config.model.checkpoint_dir}")

    @classmethod
    def from_ollama(
        cls,
        *,
        model: str = "llama3:8b",
        comparison_model: str | None = "mistral-nemo",
        config: ExtractionConfig | dict[str, Any] | str | Path | None = None,
        logger: SimpleLogger | None = None,
    ) -> "Reader":
        loaded = load_extraction_config(config)
        loaded.model.backend = "ollama"
        loaded.model.model = model
        loaded.model.comparison_model = comparison_model
        return cls(config=loaded, logger=logger)

    def run(self, source: str | Path, document_id: str | None = None) -> dict[str, Any]:
        return self.run_doc(source, document_id=document_id).to_dict()

    def run_doc(self, source: str | Path, document_id: str | None = None) -> Doc:
        document = self.loader.load(source, document_id=document_id)
        prepare_document = getattr(self.backend, "prepare_document", None)
        if callable(prepare_document):
            prepare_document(document)
        chunks = self.chunker.chunk(document)
        chunk_classification = self._classify_chunks(chunks)
        chunk_extractions = self._extract_chunks(chunks, chunk_classification)
        self._write_pass3_section_checkpoints(document, chunk_extractions)
        return self._consolidate(document, chunks, chunk_classification, chunk_extractions)

    def run_text(self, text: str, document_id: str = "inline-document") -> dict[str, Any]:
        return self.run(text, document_id=document_id)

    def _progress(
        self,
        iterable: Any,
        *,
        desc: str,
        total: int | None = None,
    ) -> Any:
        return tqdm(
            iterable,
            desc=desc,
            total=total,
            disable=not self.logger.enabled,
            file=self.logger.stream,
            leave=False,
        )

    def _build_backend(self) -> ExtractionBackend:
        backend_name = self.config.model.backend.lower()
        if backend_name == "ollama":
            return OllamaExtractionBackend(self.config)
        return HeuristicExtractionBackend()

    def _classify_chunks(self, chunks: list[Chunk]) -> list[ChunkClassification]:
        classifications = [
            self.backend.classify_chunk(chunk, self.config.sections)
            for chunk in self._progress(chunks, desc="Classifying chunks", total=len(chunks))
        ]
        routed = sum(1 for item in classifications if item.labels)
        self.logger.log("classify", f"routed {routed}/{len(chunks)} chunk(s)")
        return classifications

    def _extract_chunks(
        self,
        chunks: list[Chunk],
        classifications: list[ChunkClassification],
    ) -> list[ChunkExtraction]:
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        section_map = self.config.section_map
        outputs: list[ChunkExtraction] = []
        rounds = self.config.consolidation.consensus_rounds
        total_tasks = sum(len(classification.labels) * rounds for classification in classifications)
        self.logger.log("extract", f"scheduled {total_tasks} llm extraction pass(es)")

        with tqdm(
            total=total_tasks,
            desc="Extracting fields",
            disable=not self.logger.enabled,
            file=self.logger.stream,
            leave=False,
        ) as progress:
            for classification in classifications:
                if not classification.labels:
                    continue
                chunk = chunk_map[classification.chunk_id]
                for label in classification.labels:
                    section = section_map[label]
                    for round_index in range(rounds):
                        progress.set_postfix_str(
                            f"{chunk.chunk_id}:{label} r{round_index + 1}/{rounds}"
                        )
                        extraction = self.backend.extract_fields(
                            chunk,
                            section,
                            round_index=round_index,
                        )
                        extraction.confidence = classification.label_scores.get(
                            label,
                            classification.confidence,
                        )
                        outputs.append(extraction)
                        progress.update(1)
        self.logger.log("extract", f"filled {len(outputs)} chunk-section pass(es)")
        return outputs

    def _consolidate(
        self,
        document: RawDocument,
        chunks: list[Chunk],
        classifications: list[ChunkClassification],
        chunk_extractions: list[ChunkExtraction],
    ) -> Doc:
        field_candidates: dict[str, list[FieldCandidate]] = defaultdict(list)

        for extraction in chunk_extractions:
            for field_name, value in extraction.fields.items():
                cleaned = _normalize_space(value)
                if not cleaned:
                    continue
                candidate_spans = [
                    span
                    for span in extraction.evidence_spans
                    if span.field_name == field_name
                ]
                field_candidates[field_name].append(
                    FieldCandidate(
                        field_name=field_name,
                        value=cleaned,
                        section=extraction.section,
                        source_chunk_ids=[extraction.chunk_id],
                        evidence_spans=candidate_spans,
                        vote_weight=max(extraction.confidence, 0.1),
                        round_index=extraction.round_index,
                    )
                )

        final_fields: dict[str, str] = {}
        confidence: dict[str, float] = {}
        evidence_spans: list[EvidenceSpan] = []

        for field_name in self._progress(
            self.config.field_names,
            desc="Consolidating fields",
            total=len(self.config.field_names),
        ):
            candidates = field_candidates.get(field_name, [])
            if not candidates:
                final_fields[field_name] = ""
                confidence[field_name] = 0.0
                continue

            groups = self._group_candidates(candidates)
            selected_value = self._select_candidate_value(field_name, candidates, groups)
            selected_norm = _normalize_key(selected_value)
            selected_candidates = groups.get(selected_norm, [])
            final_fields[field_name] = selected_value
            confidence[field_name] = self._score_confidence(selected_candidates, candidates)
            evidence_spans.extend(
                self._pick_evidence_spans(field_name, selected_candidates)
            )

        section_values: dict[str, dict[str, str]] = {
            section.name: {} for section in self.config.sections
        }
        for field_name, value in final_fields.items():
            section_values[self.config.field_to_section[field_name]][field_name] = value

        metadata = {
            "backend": self.backend.name,
            "model": self.config.model.model if self.backend.name == "ollama" else "heuristic",
            "comparison_model": self.config.model.comparison_model,
            "page_count": len(document.pages),
            "chunk_count": len(chunks),
            "consensus_rounds": self.config.consolidation.consensus_rounds,
            "missing_fields": [field_name for field_name, value in final_fields.items() if not value],
        }
        self.logger.log("merge", f"consolidated {len(final_fields)} field(s)")
        return Doc(
            schema_version=self.config.schema_version,
            document_id=document.document_id,
            title=document.title,
            source_path=document.source_path,
            sections=section_values,
            fields=final_fields,
            confidence=confidence,
            evidence_spans=evidence_spans,
            chunk_classification=classifications,
            extraction_notes=[],
            metadata=metadata,
        )

    def _write_pass3_section_checkpoints(
        self,
        document: RawDocument,
        chunk_extractions: list[ChunkExtraction],
    ) -> None:
        if self.backend.name != "ollama":
            return
        checkpoint_store = getattr(self.backend, "checkpoints", None)
        if checkpoint_store is None:
            return

        section_payloads: dict[str, dict[str, Any]] = {}
        for section in self.config.sections:
            section_payloads[section.name] = {
                "schema_version": self.config.schema_version,
                "document_id": document.document_id,
                "title": document.title,
                "source_path": document.source_path,
                "section": section.name,
                "description": section.description,
                "fields": {field_config.name: [] for field_config in section.fields},
            }

        for extraction in chunk_extractions:
            section_payload = section_payloads.get(extraction.section)
            if section_payload is None:
                continue
            for field_name, value in extraction.fields.items():
                cleaned = _normalize_space(value)
                if not cleaned or _is_empty_extraction_value(cleaned):
                    continue
                field_entries = section_payload["fields"].get(field_name)
                if field_entries is None:
                    continue
                field_entries.append(
                    {
                        "value": cleaned,
                        "chunk_id": extraction.chunk_id,
                        "round_index": extraction.round_index,
                        "confidence": extraction.confidence,
                        "evidence_spans": [
                            {
                                "text": span.text,
                                "page": span.page,
                                "section": span.section,
                                "chunk_id": span.chunk_id,
                                "score": span.score,
                            }
                            for span in extraction.evidence_spans
                            if span.field_name == field_name
                        ],
                    }
                )

        output_dir = checkpoint_store.write_pass3_sections(
            document_id=document.document_id,
            title=document.title,
            section_payloads=section_payloads,
        )
        self.logger.log("pass3", f"saved accumulated section files to {output_dir}")

    def _group_candidates(
        self,
        candidates: list[FieldCandidate],
    ) -> dict[str, list[FieldCandidate]]:
        grouped: dict[str, list[FieldCandidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[_normalize_key(candidate.value)].append(candidate)
        return grouped

    def _select_candidate_value(
        self,
        field_name: str,
        candidates: list[FieldCandidate],
        groups: dict[str, list[FieldCandidate]],
    ) -> str:
        del field_name, candidates
        return self._best_group_value(groups)

    def _best_group_value(self, groups: dict[str, list[FieldCandidate]]) -> str:
        ranked = sorted(
            groups.values(),
            key=lambda items: (
                sum(candidate.vote_weight for candidate in items),
                len({span.text for candidate in items for span in candidate.evidence_spans}),
                max(len(candidate.value) for candidate in items),
            ),
            reverse=True,
        )
        return self._best_value(ranked[0])

    def _best_value(self, candidates: list[FieldCandidate]) -> str:
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                candidate.vote_weight,
                len(candidate.evidence_spans),
                len(candidate.value),
            ),
            reverse=True,
        )
        return ranked[0].value

    def _score_confidence(
        self,
        selected: list[FieldCandidate],
        all_candidates: list[FieldCandidate],
    ) -> float:
        if not selected or not all_candidates:
            return 0.0
        selected_support = sum(candidate.vote_weight for candidate in selected)
        total_support = sum(candidate.vote_weight for candidate in all_candidates)
        agreement_score = selected_support / total_support if total_support else 0.0
        unique_spans = {
            (span.text, span.page, span.chunk_id)
            for candidate in selected
            for span in candidate.evidence_spans
        }
        evidence_strength = min(
            1.0,
            len(unique_spans) / self.config.consolidation.max_evidence_spans_per_field,
        )
        supported_rounds = {candidate.round_index for candidate in selected}
        cross_pass_score = min(
            1.0,
            len(supported_rounds) / self.config.consolidation.consensus_rounds,
        )
        score = (0.3 * evidence_strength) + (0.4 * agreement_score) + (0.3 * cross_pass_score)
        return round(min(1.0, score), 2)

    def _pick_evidence_spans(
        self,
        field_name: str,
        candidates: list[FieldCandidate],
    ) -> list[EvidenceSpan]:
        seen: set[tuple[str, int | None, str | None]] = set()
        output: list[EvidenceSpan] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (item.vote_weight, len(item.evidence_spans)),
            reverse=True,
        ):
            for span in candidate.evidence_spans:
                key = (span.text, span.page, span.chunk_id)
                if key in seen:
                    continue
                seen.add(key)
                output.append(
                    EvidenceSpan(
                        field_name=field_name,
                        text=span.text,
                        section=span.section,
                        page=span.page,
                        chunk_id=span.chunk_id,
                        score=span.score,
                    )
                )
                if len(output) >= self.config.consolidation.max_evidence_spans_per_field:
                    return output
        return output


def _estimate_tokens(text: str) -> int:
    return max(1, len(WORD_RE.findall(text)))


def _normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_empty_extraction_value(text: str) -> bool:
    normalized = _normalize_key(text)
    return normalized in {"none", "null", "n/a", "na", "unknown", "not provided", "not specified"}


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ", ".join(_coerce_text(item) for item in value if _coerce_text(item))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    if value is None:
        return ""
    return str(value)


class CheckpointStore:
    """Persist every LLM request/response pair for reuse and inspection."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.legacy_root_dir = Path("debug/extraction")
        self.current_scope = ""
        self.directory_counters: dict[str, int] = {}

    def set_scope(self, document_id: str, title: str) -> None:
        self.current_scope = _safe_filename(document_id or title or "document")

    def load_cached_parsed(self, *, pass_name: str, file_stem: str) -> dict[str, Any] | None:
        safe_stem = _safe_filename(file_stem)
        for directory in self._candidate_directories(pass_name):
            if not directory.exists():
                continue
            parsed_matches = sorted(directory.glob(f"*_{safe_stem}_parsed.txt"))
            for parsed_path in reversed(parsed_matches):
                try:
                    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed

            response_matches = sorted(directory.glob(f"*_{safe_stem}_response.txt"))
            for response_path in reversed(response_matches):
                parsed = self._parsed_from_response_file(response_path)
                if parsed is not None:
                    return parsed
        return None

    def write_request(
        self,
        *,
        pass_name: str,
        file_stem: str,
        model_name: str,
        request_url: str,
        system_prompt: str,
        user_prompt: str,
        request_body: dict[str, Any],
    ) -> Path | None:
        safe_stem = _safe_filename(file_stem)
        directory = self._write_directory(pass_name)
        directory.mkdir(parents=True, exist_ok=True)
        next_index = self._next_index(directory)
        prefix = directory / f"{next_index:04d}_{safe_stem}"
        request_text = "\n".join(
            [
                f"pass: {pass_name}",
                f"model: {model_name}",
                f"url: {request_url}",
                "",
                "[system_prompt]",
                system_prompt,
                "",
                "[user_prompt]",
                user_prompt,
                "",
                "[request_json]",
                json.dumps(request_body, indent=2, ensure_ascii=True),
                "",
            ]
        )
        prefix.with_name(prefix.name + "_request.txt").write_text(request_text, encoding="utf-8")
        return prefix

    def write_response(
        self,
        prefix: Path | None,
        *,
        response_body: dict[str, Any],
        raw_response_text: str,
    ) -> None:
        if prefix is None:
            return
        response_text = "\n".join(
            [
                "[raw_response_text]",
                raw_response_text,
                "",
                "[response_json]",
                json.dumps(response_body, indent=2, ensure_ascii=True),
                "",
            ]
        )
        prefix.with_name(prefix.name + "_response.txt").write_text(response_text, encoding="utf-8")

    def write_parsed(self, prefix: Path | None, parsed: Any) -> None:
        if prefix is None:
            return
        prefix.with_name(prefix.name + "_parsed.txt").write_text(
            json.dumps(parsed, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def write_error(self, prefix: Path | None, *, message: str) -> None:
        if prefix is None:
            return
        prefix.with_name(prefix.name + "_error.txt").write_text(
            textwrap.dedent(
                f"""\
                [error]
                {message}
                """
            ),
            encoding="utf-8",
        )

    def write_pass3_sections(
        self,
        *,
        document_id: str,
        title: str,
        section_payloads: dict[str, dict[str, Any]],
    ) -> Path:
        doc_name = _safe_filename(document_id or title or "document")
        directory = self.root_dir / "pass3" / doc_name
        directory.mkdir(parents=True, exist_ok=True)
        for section_name, payload in section_payloads.items():
            path = directory / f"{_safe_filename(section_name)}.json"
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
        return directory

    def _parsed_from_response_file(self, path: Path) -> dict[str, Any] | None:
        text = path.read_text(encoding="utf-8")
        marker = "[response_json]"
        if marker not in text:
            return None
        response_json = text.split(marker, 1)[1].strip()
        try:
            payload = json.loads(response_json)
        except json.JSONDecodeError:
            return None
        content = payload.get("message", {}).get("content", "")
        try:
            parsed = _extract_json(content)
        except RuntimeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _candidate_directories(self, pass_name: str) -> list[Path]:
        directories: list[Path] = []
        if self.current_scope:
            directories.append(self.root_dir / pass_name / self.current_scope)
        directories.append(self.root_dir / pass_name)
        if self.current_scope:
            directories.append(self.legacy_root_dir / pass_name / self.current_scope)
        legacy = self.legacy_root_dir / pass_name
        if legacy not in directories:
            directories.append(legacy)
        return directories

    def _write_directory(self, pass_name: str) -> Path:
        if self.current_scope:
            return self.root_dir / pass_name / self.current_scope
        return self.root_dir / pass_name

    def _next_index(self, directory: Path) -> int:
        key = str(directory)
        if key not in self.directory_counters:
            max_existing = 0
            if directory.exists():
                for path in directory.glob("*.txt"):
                    match = re.match(r"^(\d{4})_", path.name)
                    if match:
                        max_existing = max(max_existing, int(match.group(1)))
            self.directory_counters[key] = max_existing
        self.directory_counters[key] += 1
        return self.directory_counters[key]


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "call"


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _clamp(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, number)), 2)


def _coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _extract_json(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(content):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(content[index:])
            except json.JSONDecodeError:
                continue
            return obj
    raise RuntimeError("Model output did not contain valid JSON.")
