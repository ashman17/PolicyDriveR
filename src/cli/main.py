"""CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alignment.base import Align, SimpleLogger as AlignmentLogger
from alignment.config import load_alignment_config
from extraction.base import Reader, SimpleLogger
from extraction.config import load_extraction_config
from scoring.base import Score, SimpleLogger as ScoringLogger
from scoring.config import load_scoring_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="policydriver")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("help", help="Show help")

    extract = subparsers.add_parser("extract", help="Extract structured JSON from a document")
    extract.add_argument(
        "--file",
        required=True,
        help="Path to the PDF or text file to extract",
    )
    extract.add_argument("--config", help="JSON or YAML config path", default=None)
    extract.add_argument("--output", help="Write JSON output to this file", default=None)
    extract.add_argument("--document-id", help="Override document id", default=None)
    extract.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="HTTP timeout for each Ollama call",
    )
    extract.add_argument(
        "--backend",
        choices=["heuristic", "ollama"],
        default=None,
        help="Extraction backend",
    )
    extract.add_argument("--model", help="Primary Ollama model", default=None)
    extract.add_argument("--comparison-model", help="Second-pass comparison model", default=None)
    extract.add_argument("--quiet", action="store_true", help="Suppress stage logs")

    extract_sample = subparsers.add_parser(
        "extract-sample",
        help="Extract one sample PDF from data/research",
    )
    extract_sample.add_argument(
        "--source-dir",
        default="data/research",
        help="Directory containing sample research PDFs",
    )
    extract_sample.add_argument(
        "--file",
        default=None,
        help="Specific PDF filename inside data/research",
    )
    extract_sample.add_argument("--config", help="JSON or YAML config path", default=None)
    extract_sample.add_argument("--output", help="Write JSON output to this file", default=None)
    extract_sample.add_argument("--document-id", help="Override document id", default=None)
    extract_sample.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="HTTP timeout for each Ollama call",
    )
    extract_sample.add_argument(
        "--backend",
        choices=["heuristic", "ollama"],
        default=None,
        help="Extraction backend",
    )
    extract_sample.add_argument("--model", help="Primary Ollama model", default=None)
    extract_sample.add_argument("--comparison-model", help="Second-pass comparison model", default=None)
    extract_sample.add_argument("--quiet", action="store_true", help="Suppress stage logs")

    config_cmd = subparsers.add_parser(
        "extract-config",
        help="Print the default extraction config so it can be customized",
    )
    config_cmd.add_argument("--output", help="Write config JSON to this file", default=None)

    align = subparsers.add_parser(
        "align-checkpoints",
        help="Run section-field alignment from extraction pass3 folders",
    )
    align.add_argument("--research-id", required=True, help="Research folder name under extraction pass3")
    align.add_argument(
        "--policy-id",
        action="append",
        default=None,
        help="Policy folder name under extraction pass3. Repeat to add multiple policies.",
    )
    align.add_argument(
        "--source-dir",
        default=None,
        help="Override the extraction pass3 directory used as alignment input",
    )
    align.add_argument("--config", help="JSON or YAML config path", default=None)
    align.add_argument("--output", help="Write JSON output to this file", default=None)
    align.add_argument("--model", help="Primary Ollama model", default=None)
    align.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="HTTP timeout for each Ollama alignment call",
    )
    align.add_argument("--quiet", action="store_true", help="Suppress stage logs")

    align_config_cmd = subparsers.add_parser(
        "align-config",
        help="Print the default alignment config so it can be customized",
    )
    align_config_cmd.add_argument("--output", help="Write config JSON to this file", default=None)

    score = subparsers.add_parser(
        "score-checkpoints",
        help="Score one research document across many alignment comparison files",
    )
    score.add_argument("--research-id", required=True, help="Research id used in alignment reports")
    score.add_argument(
        "--policy-id",
        action="append",
        default=None,
        help="Policy id to include. Repeat to restrict scoring to selected policies.",
    )
    score.add_argument(
        "--source-dir",
        default=None,
        help="Override the alignment final directory used as scoring input",
    )
    score.add_argument("--config", help="JSON or YAML config path", default=None)
    score.add_argument("--output", help="Write JSON output to this file", default=None)
    score.add_argument("--quiet", action="store_true", help="Suppress stage logs")

    score_config_cmd = subparsers.add_parser(
        "score-config",
        help="Print the default scoring config so it can be customized",
    )
    score_config_cmd.add_argument("--output", help="Write config JSON to this file", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command or args.command == "help":
        parser.print_help()
        return 0

    if args.command == "extract-config":
        config = load_extraction_config().to_dict()
        payload = json.dumps(config, indent=2) + "\n"
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
        return 0

    if args.command == "align-config":
        config = load_alignment_config().to_dict()
        payload = json.dumps(config, indent=2) + "\n"
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
        return 0

    if args.command == "score-config":
        config = load_scoring_config().to_dict()
        payload = json.dumps(config, indent=2) + "\n"
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
        return 0

    if args.command == "extract":
        return _run_extraction(
            source=args.file,
            config_path=args.config,
            output_path=args.output,
            document_id=args.document_id,
            timeout_seconds=args.timeout_seconds,
            backend=args.backend,
            model=args.model,
            comparison_model=args.comparison_model,
            quiet=args.quiet,
        )

    if args.command == "extract-sample":
        source_path = _resolve_sample_pdf(args.source_dir, args.file)
        output_path = args.output or _default_sample_output_path(source_path)
        return _run_extraction(
            source=str(source_path),
            config_path=args.config,
            output_path=str(output_path),
            document_id=args.document_id,
            timeout_seconds=args.timeout_seconds,
            backend=args.backend,
            model=args.model,
            comparison_model=args.comparison_model,
            quiet=args.quiet,
        )

    if args.command == "align-checkpoints":
        return _run_alignment_from_checkpoints(
            research_id=args.research_id,
            policy_ids=args.policy_id,
            source_dir=args.source_dir,
            config_path=args.config,
            output_path=args.output,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            quiet=args.quiet,
        )

    if args.command == "score-checkpoints":
        return _run_scoring_from_checkpoints(
            research_id=args.research_id,
            policy_ids=args.policy_id,
            source_dir=args.source_dir,
            config_path=args.config,
            output_path=args.output,
            quiet=args.quiet,
        )

    parser.print_help()
    return 0


def _run_extraction(
    *,
    source: str,
    config_path: str | None,
    output_path: str | None,
    document_id: str | None,
    timeout_seconds: int | None,
    backend: str | None,
    model: str | None,
    comparison_model: str | None,
    quiet: bool,
) -> int:
    try:
        config = load_extraction_config(config_path)
        if backend:
            config.model.backend = backend
        if model:
            config.model.model = model
        if comparison_model:
            config.model.comparison_model = comparison_model
        if timeout_seconds is not None:
            config.model.timeout_seconds = timeout_seconds
        reader = Reader(
            config=config,
            logger=SimpleLogger(enabled=not quiet),
        )
        result = reader.run(source, document_id=document_id)
    except Exception as exc:
        print(f"policydriver extract failed: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(result, indent=2) + "\n"
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(payload, encoding="utf-8")
        if not quiet:
            print(f"saved extraction to {output_file}", file=sys.stderr)
    else:
        if not quiet:
            print("saved section checkpoints under checkpoints/extraction/pass3", file=sys.stderr)
    return 0


def _resolve_sample_pdf(source_dir: str, filename: str | None) -> Path:
    directory = Path(source_dir)
    if not directory.exists():
        raise FileNotFoundError(
            f"Sample directory '{directory}' does not exist. Put a PDF in data/research first."
        )

    if filename:
        candidate = directory / filename
        if not candidate.exists():
            raise FileNotFoundError(f"Sample PDF '{candidate}' was not found.")
        if candidate.suffix.lower() != ".pdf":
            raise ValueError(f"Sample file '{candidate.name}' is not a PDF.")
        return candidate

    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDF files found in '{directory}'. Put one sample PDF in data/research."
        )
    return pdfs[0]


def _default_sample_output_path(source_path: Path) -> Path:
    return Path("examples/outputs") / f"{source_path.stem}_extraction.json"


def _run_alignment_from_checkpoints(
    *,
    research_id: str,
    policy_ids: list[str] | None,
    source_dir: str | None,
    config_path: str | None,
    output_path: str | None,
    model: str | None,
    timeout_seconds: int | None,
    quiet: bool,
) -> int:
    try:
        config = load_alignment_config(config_path)
        if model:
            config.model.model = model
        if timeout_seconds is not None:
            config.model.timeout_seconds = timeout_seconds
        if source_dir:
            config.input.extraction_checkpoint_dir = source_dir
        aligner = Align(
            config=config,
            logger=AlignmentLogger(enabled=not quiet),
        )
        result = aligner.run_from_checkpoints(
            research_id=research_id,
            policy_ids=policy_ids,
            source_dir=source_dir,
        )
    except Exception as exc:
        print(f"policydriver align failed: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(result, indent=2) + "\n"
    resolved_policy_ids = sorted(result.get("policy_document_ids", []))
    comparison_name = "__".join([research_id, *resolved_policy_ids]) or research_id
    safe_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in comparison_name).strip("._")
    checkpoint_path = Path(config.model.checkpoint_dir) / "final" / f"{safe_name or research_id}.json"
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(payload, encoding="utf-8")
        if not quiet:
            print(f"saved alignment to {output_file}", file=sys.stderr)
    else:
        if not quiet:
            print(f"saved final alignment checkpoint to {checkpoint_path}", file=sys.stderr)
    return 0


def _run_scoring_from_checkpoints(
    *,
    research_id: str,
    policy_ids: list[str] | None,
    source_dir: str | None,
    config_path: str | None,
    output_path: str | None,
    quiet: bool,
) -> int:
    try:
        config = load_scoring_config(config_path)
        if source_dir:
            config.input.alignment_checkpoint_dir = source_dir
        scorer = Score(
            config=config,
            logger=ScoringLogger(enabled=not quiet),
        )
        result = scorer.run_from_checkpoints(
            research_id=research_id,
            policy_ids=policy_ids,
            source_dir=source_dir,
        )
    except Exception as exc:
        print(f"policydriver score failed: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(result, indent=2) + "\n"
    checkpoint_path = Path(config.output.checkpoint_dir) / "final" / f"{research_id}.json"
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(payload, encoding="utf-8")
        if not quiet:
            print(f"saved scoring output to {output_file}", file=sys.stderr)
    else:
        if not quiet:
            print(f"saved final scoring checkpoint to {checkpoint_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
