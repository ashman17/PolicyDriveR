"""Section-field alignment layer backed by configurable rubrics."""

from __future__ import annotations

import json
import re
import socket
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from alignment.config import AlignmentConfig, RubricConfig, load_alignment_config
from alignment.prompts import load_prompt_library
from alignment.schema import AlignmentReport, FieldAlignment, PolicyEvidence

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - depends on installed extras
    def tqdm(iterable: Any | None = None, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return iterable if iterable is not None else _NullProgress()


class _NullProgress:
    def __enter__(self) -> "_NullProgress":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb

    def update(self, step: int = 1) -> None:
        del step

    def set_postfix_str(self, value: str) -> None:
        del value


class SimpleLogger:
    """Small logger with short stage-based messages."""

    def __init__(self, enabled: bool = True, stream: Any | None = None) -> None:
        self.enabled = enabled
        self.stream = stream or sys.stderr

    def log(self, stage: str, message: str) -> None:
        if not self.enabled:
            return
        print(f"[align:{stage}] {message}", file=self.stream)


class Align:
    """Compare one research extraction set against many policy extraction sets."""

    def __init__(
        self,
        config: AlignmentConfig | dict[str, Any] | str | Path | None = None,
        logger: SimpleLogger | None = None,
    ) -> None:
        self.config = load_alignment_config(config)
        self.logger = logger or SimpleLogger()
        self.prompts = load_prompt_library(self.config.prompts_path)
        self.checkpoints = CheckpointStore(Path(self.config.model.checkpoint_dir))
        self.logger.log("model", f"backend={self.config.model.backend} model={self.config.model.model}")
        self.logger.log("checkpoint", f"writing alignment checkpoints to {self.config.model.checkpoint_dir}")

    def run(
        self,
        policy_docs: list[dict[str, Any] | str | Path] | dict[str, Any] | str | Path,
        research_doc: dict[str, Any] | str | Path,
    ) -> dict[str, Any]:
        """Run alignment from already-loaded section payloads or section directories."""

        research_sections = self._coerce_section_source(research_doc)
        policy_sources = policy_docs if isinstance(policy_docs, list) else [policy_docs]
        policy_sections = {
            policy_id: payload
            for policy_id, payload in (
                self._named_section_source(source)
                for source in policy_sources
            )
        }
        report = self._run_alignment(research_sections, policy_sections)
        return report.to_dict()

    def run_from_checkpoints(
        self,
        *,
        research_id: str,
        policy_ids: list[str] | None = None,
        source_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run alignment using section JSONs written under extraction pass3."""

        base_dir = Path(source_dir or self.config.input.extraction_checkpoint_dir)
        research_sections = self._load_section_folder(base_dir / research_id)
        if policy_ids is None:
            policy_ids = [
                path.name
                for path in sorted(base_dir.glob(self.config.input.policy_folder_glob))
                if path.is_dir()
            ]
        policy_sections = {
            policy_id: self._load_section_folder(base_dir / policy_id)
            for policy_id in policy_ids
        }
        report = self._run_alignment(research_sections, policy_sections)
        return report.to_dict()

    def _run_alignment(
        self,
        research_sections: dict[str, dict[str, Any]],
        policy_sections: dict[str, dict[str, dict[str, Any]]],
    ) -> AlignmentReport:
        research_document_id = self._document_id_from_sections(research_sections)
        self.checkpoints.set_scope(research_document_id)
        tasks = self._build_tasks(research_sections, policy_sections)
        self.logger.log("tasks", f"scheduled {len(tasks)} alignment call(s)")

        field_results: list[FieldAlignment] = []
        with tqdm(
            total=len(tasks),
            desc="Aligning fields",
            disable=not self.logger.enabled,
            file=self.logger.stream,
            leave=False,
        ) as progress:
            for task in tasks:
                progress.set_postfix_str(f"{task['section']}:{task['field_name']}")
                field_results.append(self._score_field(task))
                progress.update(1)

        report = self._build_report(research_document_id, policy_sections, field_results)
        self.checkpoints.write_final(report)
        self.logger.log("final", f"saved alignment report for {research_document_id}")
        return report

    def _build_tasks(
        self,
        research_sections: dict[str, dict[str, Any]],
        policy_sections: dict[str, dict[str, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        target_map = self.config.target_map
        all_subrubrics = self.config.subrubric_names

        for section_name, section_payload in research_sections.items():
            fields = section_payload.get("fields", {})
            for field_name, raw_research_entries in fields.items():
                research_entries = self._normalize_entries(raw_research_entries)
                if not research_entries:
                    continue
                focus_subrubrics = target_map.get((section_name, field_name), [])
                policy_entries: list[dict[str, Any]] = []
                for policy_id, policy_payloads in policy_sections.items():
                    policy_field_entries = (
                        policy_payloads.get(section_name, {})
                        .get("fields", {})
                        .get(field_name, [])
                    )
                    for entry in self._normalize_entries(policy_field_entries):
                        enriched = dict(entry)
                        enriched["document_id"] = policy_id
                        policy_entries.append(enriched)
                if not policy_entries:
                    continue
                tasks.append(
                    {
                        "section": section_name,
                        "field_name": field_name,
                        "research_document_id": section_payload.get("document_id", "research"),
                        "policy_document_ids": sorted({entry["document_id"] for entry in policy_entries}),
                        "research_entries": research_entries,
                        "policy_entries": policy_entries,
                        "focus_subrubrics": focus_subrubrics,
                        "all_subrubrics": all_subrubrics,
                    }
                )
        return tasks

    def _score_field(self, task: dict[str, Any]) -> FieldAlignment:
        schema = self._comparison_schema()
        required_field_help = self._render_required_fields()
        rubric_help = self._render_rubrics()
        subrubric_help = self._render_subrubrics()
        focus_help = self._render_focus_help(task["focus_subrubrics"])
        research_entries = self._render_research_entries(task["research_entries"])
        policy_entries = self._render_policy_entries(task["policy_entries"])
        payload = self._complete_json(
            pass_name="pass1",
            file_stem=f"align_{task['section']}_{task['field_name']}_{self.config.model.model}",
            schema=schema,
            system_prompt=self.prompts.comparison.system.format(
                required_field_help=required_field_help,
                rubric_help=rubric_help,
                subrubric_help=subrubric_help,
                focus_help=focus_help,
            ),
            user_prompt=self.prompts.comparison.user.format(
                research_document_id=task["research_document_id"],
                section_name=task["section"],
                field_name=task["field_name"],
                research_entries=research_entries,
                policy_entries=policy_entries,
            ),
        )
        subrubric_scores = {
            name: _coerce_binary(payload.get("subrubric_scores", {}).get(name, 0))
            for name in self.config.subrubric_names
        }
        return FieldAlignment(
            section=task["section"],
            field_name=task["field_name"],
            research_document_id=task["research_document_id"],
            policy_document_ids=task["policy_document_ids"],
            shared_features=_clean_string_list(payload.get("shared_features", [])),
            policy_requirements_not_covered=_clean_string_list(
                payload.get("policy_requirements_not_covered", [])
            ),
            research_capabilities_not_used=_clean_string_list(
                payload.get("research_capabilities_not_used", [])
            ),
            bridge_actions=_clean_string_list(payload.get("bridge_actions", [])),
            rationale=_normalize_space(str(payload.get("rationale", ""))),
            subrubric_scores=subrubric_scores,
            research_inputs=[
                text
                for entry in task["research_entries"]
                for text in entry["evidence_texts"][:1]
            ],
            policy_inputs=[
                PolicyEvidence(document_id=entry["document_id"], text=text)
                for entry in task["policy_entries"]
                for text in entry["evidence_texts"][:1]
            ],
            metadata={
                "focus_subrubrics": task["focus_subrubrics"],
                "policy_entry_count": len(task["policy_entries"]),
                "research_entry_count": len(task["research_entries"]),
            },
        )

    def _build_report(
        self,
        research_document_id: str,
        policy_sections: dict[str, dict[str, dict[str, Any]]],
        field_results: list[FieldAlignment],
    ) -> AlignmentReport:
        policy_document_ids = sorted(policy_sections)
        subrubric_scores = {
            subrubric.name: max(
                (result.subrubric_scores.get(subrubric.name, 0) for result in field_results),
                default=0,
            )
            for subrubric in self.config.all_subrubrics
        }
        dimension_scores: dict[str, float] = {}
        for rubric in self.config.rubrics:
            dimension_scores[rubric.name] = round(
                sum(
                    subrubric.weight * subrubric_scores.get(subrubric.name, 0)
                    for subrubric in rubric.subrubrics
                ),
                2,
            )
        overall_score = round(
            sum(rubric.weight * dimension_scores.get(rubric.name, 0.0) for rubric in self.config.rubrics),
            2,
        )
        rationales = _unique_preserve_order(
            [result.rationale for result in field_results if result.rationale]
        )
        return AlignmentReport(
            schema_version=self.config.schema_version,
            research_document_id=research_document_id,
            policy_document_ids=policy_document_ids,
            field_results=field_results,
            shared_features=_flatten_unique_lists(result.shared_features for result in field_results),
            policy_requirements_not_covered=_flatten_unique_lists(
                result.policy_requirements_not_covered for result in field_results
            ),
            research_capabilities_not_used=_flatten_unique_lists(
                result.research_capabilities_not_used for result in field_results
            ),
            bridge_actions=_flatten_unique_lists(result.bridge_actions for result in field_results),
            rationale=" ".join(rationales[:3]),
            subrubric_scores=subrubric_scores,
            dimension_scores=dimension_scores,
            overall_score=overall_score,
            overall_percent=int(round(overall_score * 100)),
            metadata={
                "model": self.config.model.model,
                "policy_count": len(policy_document_ids),
                "field_call_count": len(field_results),
            },
        )

    def _comparison_schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for output_field in self.config.output_fields:
            if output_field.kind == "string":
                properties[output_field.name] = {"type": "string"}
            else:
                properties[output_field.name] = {
                    "type": "array",
                    "items": {"type": "string"},
                }
            required.append(output_field.name)

        properties["subrubric_scores"] = {
            "type": "object",
            "properties": {
                subrubric.name: {"type": "integer", "enum": [0, 1]}
                for subrubric in self.config.all_subrubrics
            },
            "required": [subrubric.name for subrubric in self.config.all_subrubrics],
        }
        required.append("subrubric_scores")
        return {"type": "object", "properties": properties, "required": required}

    def _render_required_fields(self) -> str:
        lines = []
        for output_field in self.config.output_fields:
            lines.append(f"- {output_field.name}: {output_field.description}")
        lines.append("- subrubric_scores: binary 0/1 scores for every subrubric")
        return "\n".join(lines)

    def _render_rubrics(self) -> str:
        return "\n".join(
            f"- {rubric.name} (weight={rubric.weight:.2f}): {rubric.description}"
            for rubric in self.config.rubrics
        )

    def _render_subrubrics(self) -> str:
        lines = []
        for rubric in self.config.rubrics:
            lines.append(f"{rubric.name}:")
            for subrubric in rubric.subrubrics:
                lines.append(
                    f"- {subrubric.name} (weight={subrubric.weight:.2f}): {subrubric.description}"
                )
        return "\n".join(lines)

    def _render_focus_help(self, focus_subrubrics: list[str]) -> str:
        if not focus_subrubrics:
            return "- No special focus cues for this field."
        return "\n".join(f"- {name}" for name in focus_subrubrics)

    def _render_research_entries(self, entries: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for index, entry in enumerate(entries, start=1):
            lines.append(f"{index}. value={entry['value']!r}")
            lines.append("   exact evidence:")
            for evidence in entry["evidence_texts"]:
                lines.append(f"   - {evidence!r}")
        return "\n".join(lines)

    def _render_policy_entries(self, entries: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for index, entry in enumerate(entries, start=1):
            lines.append(f"{index}. document_id={entry['document_id']}; value={entry['value']!r}")
            lines.append("   exact evidence:")
            for evidence in entry["evidence_texts"]:
                lines.append(f"   - {evidence!r}")
        return "\n".join(lines)

    def _complete_json(
        self,
        *,
        pass_name: str,
        file_stem: str,
        schema: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        url = f"{self.config.model.base_url.rstrip('/')}/api/chat"
        body = {
            "model": self.config.model.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
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
            model_name=self.config.model.model,
            request_url=url,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            request_body=body,
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.model.timeout_seconds) as response:
                raw_response_text = response.read().decode("utf-8")
                payload = json.loads(raw_response_text)
        except urllib.error.HTTPError as exc:  # pragma: no cover - runtime dependent
            error_body = exc.read().decode("utf-8", errors="replace")
            self.checkpoints.write_error(
                checkpoint_prefix,
                message=f"HTTPError {exc.code}: {exc.reason}\n\n{error_body}",
            )
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code} during alignment. "
                f"See checkpoint files under {self.config.model.checkpoint_dir}/{pass_name}."
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            self.checkpoints.write_error(
                checkpoint_prefix,
                message=(
                    f"Timeout after {self.config.model.timeout_seconds}s during alignment "
                    f"for model {self.config.model.model}: {exc}"
                ),
            )
            raise RuntimeError(
                f"Ollama request timed out after {self.config.model.timeout_seconds}s during alignment."
            ) from exc
        except urllib.error.URLError as exc:  # pragma: no cover - runtime dependent
            self.checkpoints.write_error(
                checkpoint_prefix,
                message=f"{type(exc).__name__}: {exc}",
            )
            raise RuntimeError(
                "Could not reach Ollama for alignment. Start Ollama locally or point base_url to the correct hosted endpoint."
            ) from exc

        self.checkpoints.write_response(
            checkpoint_prefix,
            response_body=payload,
            raw_response_text=raw_response_text,
        )
        parsed = _extract_json(payload.get("message", {}).get("content", ""))
        self.checkpoints.write_parsed(checkpoint_prefix, parsed)
        if not isinstance(parsed, dict):
            raise RuntimeError("Alignment model returned non-object JSON.")
        return parsed

    def _coerce_section_source(self, source: dict[str, Any] | str | Path) -> dict[str, dict[str, Any]]:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                return self._load_section_folder(path)
            raise ValueError(f"Alignment source path '{path}' is not a directory.")
        if "section" in source and "fields" in source:
            return {str(source["section"]): source}
        return {str(key): value for key, value in source.items()}

    def _named_section_source(
        self,
        source: dict[str, Any] | str | Path,
    ) -> tuple[str, dict[str, dict[str, Any]]]:
        payload = self._coerce_section_source(source)
        return self._document_id_from_sections(payload), payload

    def _load_section_folder(self, directory: Path) -> dict[str, dict[str, Any]]:
        if not directory.exists():
            raise FileNotFoundError(f"Section folder '{directory}' was not found.")
        section_payloads: dict[str, dict[str, Any]] = {}
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            section_name = str(payload.get("section", path.stem))
            section_payloads[section_name] = payload
        if not section_payloads:
            raise ValueError(f"No section JSON files found in '{directory}'.")
        return section_payloads

    def _document_id_from_sections(self, sections: dict[str, dict[str, Any]]) -> str:
        for payload in sections.values():
            document_id = _normalize_space(str(payload.get("document_id", "")))
            if document_id:
                return document_id
        return "document"

    def _normalize_entries(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for entry in entries or []:
            value = _normalize_space(str(entry.get("value", "")))
            if not value or _is_empty_value(value):
                continue
            evidence_texts = _clean_string_list(
                span.get("text", "")
                for span in entry.get("evidence_spans", [])
            )
            if not evidence_texts:
                evidence_texts = [value]
            output.append(
                {
                    "value": value,
                    "confidence": float(entry.get("confidence", 0.0) or 0.0),
                    "evidence_texts": evidence_texts,
                }
            )
        return output


class CheckpointStore:
    """Persist alignment request/response pairs and final reports."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.current_scope = ""
        self.directory_counters: dict[str, int] = {}

    def set_scope(self, research_document_id: str) -> None:
        self.current_scope = _safe_filename(research_document_id or "research")

    def load_cached_parsed(self, *, pass_name: str, file_stem: str) -> dict[str, Any] | None:
        safe_stem = _safe_filename(file_stem)
        directory = self._write_directory(pass_name)
        if not directory.exists():
            return None
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
    ) -> Path:
        safe_stem = _safe_filename(file_stem)
        directory = self._write_directory(pass_name)
        directory.mkdir(parents=True, exist_ok=True)
        prefix = directory / f"{self._next_index(directory):04d}_{safe_stem}"
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

    def write_final(self, report: AlignmentReport) -> Path:
        directory = self.root_dir / "final"
        directory.mkdir(parents=True, exist_ok=True)
        filename = _safe_filename(report.research_document_id or "research")
        path = directory / f"{filename}.json"
        path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        return path

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
        try:
            parsed = _extract_json(payload.get("message", {}).get("content", ""))
        except RuntimeError:
            return None
        return parsed if isinstance(parsed, dict) else None

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


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_key(text: str) -> str:
    return _normalize_space(text).lower()


def _is_empty_value(text: str) -> bool:
    return _normalize_key(text) in {"", "none", "null", "n/a", "na", "unknown", "not provided", "not specified"}


def _clean_string_list(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    output: list[str] = []
    for value in values or []:
        cleaned = _normalize_space(str(value))
        if not cleaned or _is_empty_value(cleaned):
            continue
        output.append(cleaned)
    return _unique_preserve_order(output)


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _flatten_unique_lists(groups: Any) -> list[str]:
    flattened: list[str] = []
    for group in groups:
        flattened.extend(group)
    return _unique_preserve_order(flattened)


def _coerce_binary(value: Any) -> int:
    if value in {1, "1", True}:
        return 1
    return 0


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
