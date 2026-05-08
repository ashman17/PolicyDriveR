"""Final scoring and normalization layer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from alignment.config import load_alignment_config
from scoring.config import ScoringConfig, load_scoring_config
from scoring.schema import RubricScore, ScoreReport, SubrubricScore


class SimpleLogger:
    """Small logger with short stage-based messages."""

    def __init__(self, enabled: bool = True, stream: Any | None = None) -> None:
        self.enabled = enabled
        self.stream = stream or sys.stderr

    def log(self, stage: str, message: str) -> None:
        if not self.enabled:
            return
        print(f"[score:{stage}] {message}", file=self.stream)


class Score:
    """Aggregate many alignment comparisons into one normalized scorecard."""

    def __init__(
        self,
        config: ScoringConfig | dict[str, Any] | str | Path | None = None,
        logger: SimpleLogger | None = None,
    ) -> None:
        self.config = load_scoring_config(config)
        self.logger = logger or SimpleLogger()
        self.alignment_config = load_alignment_config(self.config.alignment_config_path)
        self.logger.log("config", f"reading alignment reports from {self.config.input.alignment_checkpoint_dir}")

    def run(
        self,
        reports: list[dict[str, Any]],
        *,
        research_document_id: str | None = None,
    ) -> dict[str, Any]:
        report = self._score_reports(reports, research_document_id=research_document_id)
        self._write_final(report)
        return report.to_dict()

    def run_from_checkpoints(
        self,
        *,
        research_id: str,
        policy_ids: list[str] | None = None,
        source_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        reports = self._load_reports(
            research_id=research_id,
            policy_ids=policy_ids,
            source_dir=source_dir,
        )
        report = self._score_reports(reports, research_document_id=research_id)
        self._write_final(report)
        return report.to_dict()

    def _load_reports(
        self,
        *,
        research_id: str,
        policy_ids: list[str] | None,
        source_dir: str | Path | None,
    ) -> list[dict[str, Any]]:
        base_dir = Path(source_dir or self.config.input.alignment_checkpoint_dir)
        reports: list[dict[str, Any]] = []
        if policy_ids:
            for policy_id in policy_ids:
                path = base_dir / f"{research_id}__{policy_id}.json"
                if not path.exists():
                    raise FileNotFoundError(f"Alignment report '{path}' was not found.")
                reports.append(json.loads(path.read_text(encoding="utf-8")))
            return reports

        for path in sorted(base_dir.glob(self.config.input.comparison_glob)):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("research_document_id") != research_id:
                continue
            policy_document_ids = payload.get("policy_document_ids", [])
            if len(policy_document_ids) != 1:
                continue
            reports.append(payload)
        if not reports:
            raise FileNotFoundError(
                f"No single-policy alignment reports found for '{research_id}' in '{base_dir}'."
            )
        return reports

    def _score_reports(
        self,
        reports: list[dict[str, Any]],
        *,
        research_document_id: str | None,
    ) -> ScoreReport:
        if not reports:
            raise ValueError("Scoring requires at least one alignment report.")

        research_id = research_document_id or str(reports[0].get("research_document_id", "research"))
        policy_ids = sorted(
            {
                policy_id
                for report in reports
                for policy_id in report.get("policy_document_ids", [])
            }
        )
        subrubric_details: list[SubrubricScore] = []
        subrubric_scores: dict[str, int] = {}

        for subrubric in self.alignment_config.all_subrubrics:
            detail = self._score_subrubric(subrubric.name, reports)
            subrubric_details.append(detail)
            subrubric_scores[subrubric.name] = detail.score

        rubric_details: list[RubricScore] = []
        rubric_scores: dict[str, int] = {}
        rubric_rates: dict[str, float] = {}
        total_rubric_weight = sum(rubric.weight for rubric in self.alignment_config.rubrics) or 1.0
        for rubric in self.alignment_config.rubrics:
            weight_total = sum(subrubric.weight for subrubric in rubric.subrubrics) or 1.0
            rate = sum(
                (subrubric_scores[subrubric.name] / 100.0) * subrubric.weight
                for subrubric in rubric.subrubrics
            ) / weight_total
            score = int(round(rate * 100))
            rubric_details.append(
                RubricScore(
                    name=rubric.name,
                    score=score,
                    normalized_rate=round(rate, 4),
                    weight=rubric.weight,
                    subrubric_scores={sub.name: subrubric_scores[sub.name] for sub in rubric.subrubrics},
                )
            )
            rubric_scores[rubric.name] = score
            rubric_rates[rubric.name] = rate

        overall_rate = sum(
            rubric_rates[rubric.name] * rubric.weight
            for rubric in self.alignment_config.rubrics
        ) / total_rubric_weight
        overall_score = int(round(overall_rate * 100))

        self.logger.log("score", f"scored {research_id} across {len(policy_ids)} policy comparison(s)")
        return ScoreReport(
            schema_version=self.config.schema_version,
            research_document_id=research_id,
            policy_document_ids=policy_ids,
            subrubric_scores=subrubric_scores,
            rubric_scores=rubric_scores,
            overall_score=overall_score,
            overall_rate=round(overall_rate, 4),
            subrubric_details=subrubric_details,
            rubric_details=rubric_details,
            metadata={
                "normalization": {
                    "prior_rate": self.config.normalization.prior_rate,
                    "prior_strength": self.config.normalization.prior_strength,
                    "pair_weight_mode": self.config.normalization.pair_weight_mode,
                    "focus_metadata_key": self.config.normalization.focus_metadata_key,
                },
                "report_count": len(reports),
            },
        )

    def _score_subrubric(self, subrubric_name: str, reports: list[dict[str, Any]]) -> SubrubricScore:
        focus_key = self.config.normalization.focus_metadata_key
        pair_values: list[tuple[float, float, str]] = []
        applicable_policy_ids: list[str] = []
        total_field_count = 0

        for report in reports:
            field_results = report.get("field_results", [])
            applicable_fields = [
                field_result
                for field_result in field_results
                if subrubric_name in field_result.get("metadata", {}).get(focus_key, [])
            ]
            if not applicable_fields:
                continue
            policy_id = next(iter(report.get("policy_document_ids", [])), "policy")
            applicable_policy_ids.append(policy_id)
            field_values = [
                int(field_result.get("subrubric_scores", {}).get(subrubric_name, 0))
                for field_result in applicable_fields
            ]
            pair_rate = sum(field_values) / len(field_values)
            pair_weight = self._pair_weight(len(applicable_fields))
            pair_values.append((pair_rate, pair_weight, policy_id))
            total_field_count += len(applicable_fields)

        raw_success = sum(rate * weight for rate, weight, _ in pair_values)
        raw_weight = sum(weight for _, weight, _ in pair_values)
        raw_rate = raw_success / raw_weight if raw_weight else self.config.normalization.prior_rate

        prior_rate = self.config.normalization.prior_rate
        prior_strength = self.config.normalization.prior_strength
        normalized_rate = (raw_success + (prior_rate * prior_strength)) / (raw_weight + prior_strength)
        score = int(round(normalized_rate * 100))
        return SubrubricScore(
            name=subrubric_name,
            score=score,
            raw_rate=round(raw_rate, 4),
            normalized_rate=round(normalized_rate, 4),
            applicable_policy_count=len(applicable_policy_ids),
            applicable_field_count=total_field_count,
            policy_document_ids=sorted(set(applicable_policy_ids)),
        )

    def _pair_weight(self, applicable_field_count: int) -> float:
        mode = self.config.normalization.pair_weight_mode
        if mode == "equal":
            return 1.0
        return float(max(1, applicable_field_count))

    def _write_final(self, report: ScoreReport) -> None:
        directory = Path(self.config.output.checkpoint_dir) / "final"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{report.research_document_id}.json"
        path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

