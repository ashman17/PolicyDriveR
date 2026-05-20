from __future__ import annotations

import json

import cli.main as cli_main


def test_override_ollama_port_rewrites_default_base_url() -> None:
    assert cli_main._override_ollama_port("http://localhost:11434", 22456) == "http://localhost:22456"


def test_override_ollama_port_preserves_path() -> None:
    assert (
        cli_main._override_ollama_port("http://127.0.0.1:11434/proxy/ollama", 18080)
        == "http://127.0.0.1:18080/proxy/ollama"
    )


def test_run_extraction_applies_ollama_port_override(monkeypatch, tmp_path) -> None:
    input_path = tmp_path / "research.txt"
    input_path.write_text("Example research content.", encoding="utf-8")
    captured: dict[str, str] = {}

    class FakeReader:
        def __init__(self, config, logger) -> None:
            captured["base_url"] = config.model.base_url

        def run(self, source: str, document_id: str | None = None) -> dict[str, object]:
            return {"document_id": document_id or "research", "source": source}

    monkeypatch.setattr(cli_main, "Reader", FakeReader)

    code = cli_main._run_extraction(
        source=str(input_path),
        config_path=None,
        output_path=str(tmp_path / "out.json"),
        document_id="research1",
        timeout_seconds=None,
        backend=None,
        model=None,
        comparison_model=None,
        ollama_port=22456,
        quiet=True,
    )

    assert code == 0
    assert captured["base_url"] == "http://localhost:22456"
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert payload["document_id"] == "research1"


def test_run_alignment_applies_ollama_port_override(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakeAlign:
        def __init__(self, config, logger) -> None:
            captured["base_url"] = config.model.base_url
            self.config = config

        def run_from_checkpoints(self, research_id: str, policy_ids, source_dir=None) -> dict[str, object]:
            return {
                "research_document_id": research_id,
                "policy_document_ids": list(policy_ids or []),
                "field_results": [],
            }

    monkeypatch.setattr(cli_main, "Align", FakeAlign)

    code = cli_main._run_alignment_from_checkpoints(
        research_id="research1",
        policy_ids=["policy4"],
        source_dir=None,
        config_path=None,
        output_path=None,
        model=None,
        timeout_seconds=None,
        ollama_port=22456,
        quiet=True,
    )

    assert code == 0
    assert captured["base_url"] == "http://localhost:22456"
