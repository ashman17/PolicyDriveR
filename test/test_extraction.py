import json

from quad.cli.main import main
from quad.extraction.base import Reader, SimpleLogger
from quad.extraction.config import load_extraction_config


SAMPLE_TEXT = """
Endangered Marine Species Classification for Coastal Policy

Problem: Researchers and regulators lack a reliable way to classify endangered marine species from underwater imagery at scale.
Primary goal: Build a classification workflow that supports biodiversity monitoring and policy decisions.
Secondary goals: Improve habitat reporting and support marine reserve planning.

Methods: We use a computer vision classification pipeline with a CNN classifier and transfer learning.
Techniques: Image augmentation, confidence thresholding, and manual review.
Data sources: Reef survey photographs, NGO biodiversity archives, and government monitoring records.
Model details: The model is a CNN tuned for low-light underwater imagery.

Results: Outputs include a labeled dataset, a trained model, and policy recommendations for marine monitoring teams.
Evaluation metrics: Accuracy, F1 score, and recall are reported for species-level classification.

Constraints: Legal constraints include Endangered Species Act compliance and permit rules for protected waters.
Ethical considerations: Human review is required before enforcement use to reduce ecological harm.
Data governance: Access to location data is restricted and shared under stewardship agreements.
Deployment constraints: Limited onboard compute and intermittent connectivity constrain field deployment.

Stakeholders: Actors include government agencies, academia, and NGOs.
Beneficiaries: Coastal communities and conservation officers benefit from faster monitoring.

Deployment: The current deployment stage is prototype.
Scalability notes: The workflow can scale if agencies share annotated imagery and edge hardware improves.
""".strip()


def test_reader_extracts_configured_fields_with_heuristic_backend() -> None:
    config = load_extraction_config({"model": {"backend": "heuristic"}})
    reader = Reader(config=config, logger=SimpleLogger(enabled=False))

    result = reader.run_text(SAMPLE_TEXT, document_id="marine-paper")

    assert result["document_id"] == "marine-paper"
    assert result["fields"]["problem_statement"]
    assert result["fields"]["method_type"]
    assert result["fields"]["outputs"]
    assert result["fields"]["deployment_stage"] == "prototype"
    assert result["confidence"]["problem_statement"] > 0
    assert any(span["field_name"] == "problem_statement" for span in result["evidence_spans"])
    assert any("problem" in item["labels"] for item in result["chunk_classification"])


def test_custom_config_can_add_new_field(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {"backend": "heuristic"},
                "sections": [
                    {
                        "name": "policy",
                        "description": "Policy-specific hooks.",
                        "keywords": ["policy", "compliance"],
                        "fields": [
                            {
                                "name": "policy_hook",
                                "description": "Most important policy linkage.",
                                "keywords": ["policy hook", "compliance"],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    reader = Reader(config=config_path, logger=SimpleLogger(enabled=False))
    result = reader.run_text(
        "Policy hook: Endangered Species Act compliance is a core requirement.",
        document_id="policy-inline",
    )

    assert result["fields"]["policy_hook"] == "Endangered Species Act compliance is a core requirement."
    assert result["confidence"]["policy_hook"] > 0


def test_cli_extract_works_with_explicit_backend_override(tmp_path, capsys) -> None:
    input_path = tmp_path / "research.txt"
    input_path.write_text(SAMPLE_TEXT, encoding="utf-8")

    code = main(["extract", str(input_path), "--backend", "heuristic", "--quiet"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 0
    assert payload["document_id"] == "research"
    assert payload["metadata"]["backend"] == "heuristic"
