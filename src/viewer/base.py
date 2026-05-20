"""Static HTML dashboard rendering for research-to-policy outputs."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from alignment.config import load_alignment_config
from scoring.base import Score, SimpleLogger as ScoringLogger
from scoring.config import load_scoring_config
from viewer.config import ViewerConfig, load_viewer_config


class SimpleLogger:
    """Small logger with short stage-based messages."""

    def __init__(self, enabled: bool = True, stream: Any | None = None) -> None:
        self.enabled = enabled
        self.stream = stream or sys.stderr

    def log(self, stage: str, message: str) -> None:
        if not self.enabled:
            return
        print(f"[viewer:{stage}] {message}", file=self.stream)


class Viewer:
    """Render one combined dashboard per research document."""

    def __init__(
        self,
        config: ViewerConfig | dict[str, Any] | str | Path | None = None,
        logger: SimpleLogger | None = None,
    ) -> None:
        self.config = load_viewer_config(config)
        self.logger = logger or SimpleLogger()
        self.alignment_config = load_alignment_config(self.config.alignment_config_path)

    def default_output_path(self, research_id: str) -> Path:
        safe_name = _safe_slug(research_id) or "research"
        return Path(self.config.output.dashboard_dir) / f"{safe_name}.html"

    def run_from_checkpoints(
        self,
        *,
        research_id: str,
        alignment_source_dir: str | Path | None = None,
        scoring_source_dir: str | Path | None = None,
        output_path: str | Path | None = None,
        title: str | None = None,
    ) -> str:
        alignment_reports = self._load_alignment_reports(
            research_id=research_id,
            source_dir=alignment_source_dir,
        )
        score_report = self._resolve_score_report(
            research_id=research_id,
            alignment_reports=alignment_reports,
            alignment_source_dir=alignment_source_dir,
            scoring_source_dir=scoring_source_dir,
        )
        self.logger.log(
            "render",
            f"building combined dashboard for {research_id} across {len(alignment_reports)} policy comparison(s)",
        )
        return self._render_dashboard(
            research_id=research_id,
            alignment_reports=alignment_reports,
            score_report=score_report,
            output_path=Path(output_path) if output_path else self.default_output_path(research_id),
            title=title or self.config.ui.title,
        )

    def _load_alignment_reports(
        self,
        *,
        research_id: str,
        source_dir: str | Path | None,
    ) -> list[dict[str, Any]]:
        base_dir = Path(source_dir or self.config.input.alignment_checkpoint_dir)
        reports: list[dict[str, Any]] = []
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

    def _resolve_score_report(
        self,
        *,
        research_id: str,
        alignment_reports: list[dict[str, Any]],
        alignment_source_dir: str | Path | None,
        scoring_source_dir: str | Path | None,
    ) -> dict[str, Any]:
        base_dir = Path(scoring_source_dir or self.config.input.scoring_checkpoint_dir)
        path = base_dir / f"{research_id}.json"
        aligned_policy_ids = sorted(
            {
                policy_id
                for report in alignment_reports
                for policy_id in report.get("policy_document_ids", [])
            }
        )
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if sorted(payload.get("policy_document_ids", [])) == aligned_policy_ids:
                self.logger.log("score", f"using cached score report at {path}")
                return payload

        self.logger.log("score", "recomputing score report to match the full policy set")
        scoring_config = load_scoring_config(self.config.scoring_config_path)
        if alignment_source_dir:
            scoring_config.input.alignment_checkpoint_dir = str(alignment_source_dir)
        scorer = Score(
            config=scoring_config,
            logger=ScoringLogger(enabled=False),
        )
        return scorer._score_reports(  # noqa: SLF001 - deliberate internal reuse to avoid extra checkpoint writes
            alignment_reports,
            research_document_id=research_id,
        ).to_dict()

    def _render_dashboard(
        self,
        *,
        research_id: str,
        alignment_reports: list[dict[str, Any]],
        score_report: dict[str, Any],
        output_path: Path,
        title: str,
    ) -> str:
        policy_ids = sorted(
            {
                policy_id
                for report in alignment_reports
                for policy_id in report.get("policy_document_ids", [])
            }
        )
        rubric_detail_map = {
            detail["name"]: detail
            for detail in score_report.get("rubric_details", [])
        }
        subrubric_detail_map = {
            detail["name"]: detail
            for detail in score_report.get("subrubric_details", [])
        }
        top_rubrics = sorted(
            score_report.get("rubric_scores", {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )
        strongest = top_rubrics[:5]
        weakest = list(reversed(top_rubrics[-5:])) if top_rubrics else []
        highlights = {
            "shared_features": self._aggregate_insights(alignment_reports, "shared_features"),
            "policy_requirements_not_covered": self._aggregate_insights(
                alignment_reports,
                "policy_requirements_not_covered",
            ),
            "research_capabilities_not_used": self._aggregate_insights(
                alignment_reports,
                "research_capabilities_not_used",
            ),
            "bridge_actions": self._aggregate_insights(alignment_reports, "bridge_actions"),
        }
        policy_panels = [
            self._render_policy_panel(report)
            for report in sorted(
                alignment_reports,
                key=lambda item: item.get("policy_document_ids", ["policy"])[0],
            )
        ]
        document_links = self._build_document_links(
            document_ids=[research_id, *policy_ids],
            output_path=output_path,
        )
        document_links_json = json.dumps(document_links, sort_keys=True)
        chunk_text_map = self._build_chunk_text_map([research_id, *policy_ids])
        chunk_text_map_json = json.dumps(chunk_text_map, sort_keys=True)
        research_tag = self._render_document_tag(research_id, variant="meta")
        policy_set_tags = "".join(self._render_document_tag(policy_id, variant="meta") for policy_id in policy_ids)
        ordered_rubrics = sorted(
            self.alignment_config.rubrics,
            key=lambda rubric: rubric.name == "risk_alignment",
        )
        rubric_cards = [
            self._render_rubric_card(
                rubric,
                rubric_detail_map.get(rubric.name, {}),
                subrubric_detail_map,
            )
            for rubric in ordered_rubrics
        ]
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        executive_summary = self._render_executive_summary(
            overall_score=int(score_report.get("overall_score", 0)),
            policy_count=len(policy_ids),
            strongest=strongest,
            weakest=weakest,
            bridge_actions=highlights["bridge_actions"],
        )
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)} - {escape(research_id)}</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --paper: #fffaf2;
      --ink: #131313;
      --muted: #5f5a53;
      --line: #1e1b18;
      --card: #fef8ee;
      --strong: #101418;
      --accent: #0f766e;
      --accent-soft: #d8f0ea;
      --alert: #8a241b;
      --warm: #b45309;
      --shadow: 0 18px 48px rgba(22, 18, 13, 0.12);
      --radius: 24px;
      --panel-width: minmax(320px, 30vw);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.08), transparent 26%),
        radial-gradient(circle at top right, rgba(180, 83, 9, 0.1), transparent 30%),
        linear-gradient(180deg, #f7f2e8 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
      line-height: 1.45;
    }}
    .app-shell {{
      width: 100%;
      min-height: 100vh;
      display: grid;
      grid-template-areas: "main panel";
      grid-template-columns: minmax(0, 1fr) 0;
      transition: grid-template-columns 240ms ease;
      align-items: start;
    }}
    .app-shell.viewer-open {{
      grid-template-columns: minmax(0, 1fr) var(--panel-width);
    }}
    .pdf-panel {{
      grid-area: panel;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: hidden;
      border-right: 1px solid rgba(30, 27, 24, 0.12);
      background:
        linear-gradient(180deg, rgba(21, 21, 21, 0.98), rgba(39, 49, 58, 0.98)),
        #172027;
      color: #f5efe6;
      transform: translateX(100%);
      opacity: 0;
      pointer-events: none;
      transition: transform 240ms ease, opacity 240ms ease;
    }}
    .pdf-resizer {{
      position: absolute;
      top: 0;
      left: -7px;
      width: 14px;
      height: 100%;
      cursor: col-resize;
      z-index: 2;
      touch-action: none;
    }}
    .pdf-resizer::before {{
      content: "";
      position: absolute;
      top: 0;
      left: 6px;
      width: 2px;
      height: 100%;
      background: rgba(255, 250, 242, 0.16);
      transition: background 160ms ease;
    }}
    .pdf-resizer:hover::before,
    .pdf-resizer.dragging::before {{
      background: rgba(255, 250, 242, 0.45);
    }}
    .app-shell.viewer-open .pdf-panel {{
      transform: translateX(0);
      opacity: 1;
      pointer-events: auto;
    }}
    .pdf-panel-inner {{
      height: 100%;
      display: grid;
      grid-template-rows: auto auto 1fr;
    }}
    .pdf-panel-head {{
      padding: 18px 18px 12px;
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid rgba(255, 250, 242, 0.12);
    }}
    .pdf-panel-copy {{
      display: grid;
      gap: 6px;
      min-width: 0;
    }}
    .pdf-panel-copy h2 {{
      margin: 0;
      font-size: 1.1rem;
      line-height: 1.1;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
    }}
    .pdf-panel-copy p {{
      margin: 0;
      color: rgba(245, 239, 230, 0.76);
      font-size: 0.92rem;
    }}
    .pdf-panel-actions {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-shrink: 0;
    }}
    .pdf-action,
    .pdf-close {{
      appearance: none;
      border: 1px solid rgba(255, 250, 242, 0.22);
      border-radius: 999px;
      background: rgba(255, 250, 242, 0.08);
      color: #f5efe6;
      padding: 8px 12px;
      font: inherit;
      font-size: 0.82rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-decoration: none;
      cursor: pointer;
      transition: background 160ms ease, border-color 160ms ease;
    }}
    .pdf-action:hover,
    .pdf-close:hover {{
      background: rgba(255, 250, 242, 0.16);
      border-color: rgba(255, 250, 242, 0.4);
    }}
    .pdf-panel-status {{
      padding: 12px 18px;
      border-bottom: 1px solid rgba(255, 250, 242, 0.1);
      color: rgba(245, 239, 230, 0.82);
      font-size: 0.88rem;
    }}
    .pdf-viewer {{
      position: relative;
      height: 100%;
      overflow: auto;
      padding: 18px;
      background: linear-gradient(180deg, #243039, #1b242c);
    }}
    .pdf-stage {{
      position: relative;
      width: 100%;
      min-height: 100%;
      display: grid;
      align-content: start;
      justify-items: center;
      gap: 18px;
    }}
    .pdf-page-wrap {{
      position: relative;
      width: fit-content;
      max-width: 100%;
      box-shadow: 0 18px 48px rgba(0, 0, 0, 0.24);
      background: #ffffff;
    }}
    .pdf-canvas {{
      display: block;
      max-width: 100%;
      height: auto;
      background: #ffffff;
    }}
    .pdf-highlight-layer {{
      position: absolute;
      inset: 0;
      pointer-events: none;
    }}
    .pdf-highlight {{
      position: absolute;
      border-radius: 4px;
      background: rgba(255, 214, 10, 0.35);
      outline: 1px solid rgba(255, 214, 10, 0.8);
      box-shadow: 0 0 0 1px rgba(120, 85, 0, 0.1);
    }}
    .shell {{
      grid-area: main;
      width: min(1760px, calc(100% - 24px));
      margin: 0 auto;
      padding: 28px 0 56px;
      min-width: 0;
    }}
    .hero {{
      background: linear-gradient(145deg, #151515, #27313a);
      color: #f5efe6;
      border: 2px solid #0d0d0d;
      border-radius: 30px;
      padding: 28px;
      box-shadow: var(--shadow);
      display: grid;
      gap: 18px;
      grid-template-columns: 1.3fr 0.8fr;
    }}
    .hero h1,
    .section-title,
    .rubric-card h3,
    .policy-card h3,
    .insight-card h3,
    .metric-label,
    summary {{
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
    }}
    .hero h1 {{
      margin: 0;
      font-size: clamp(2rem, 3.6vw, 3.6rem);
      line-height: 0.98;
      letter-spacing: -0.04em;
    }}
    .hero p {{
      margin: 0;
      max-width: 70ch;
      color: rgba(245, 239, 230, 0.86);
      font-size: 1rem;
    }}
    .meta-grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }}
    .meta-chip {{
      background: rgba(255, 250, 242, 0.08);
      border: 1px solid rgba(255, 250, 242, 0.18);
      border-radius: 18px;
      padding: 14px 16px;
    }}
    .meta-chip strong {{
      display: block;
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: rgba(245, 239, 230, 0.7);
      margin-bottom: 8px;
    }}
    .meta-chip span {{
      font-size: 1rem;
      font-weight: 700;
    }}
    .section {{
      margin-top: 24px;
      background: rgba(255, 250, 242, 0.55);
      border: 2px solid rgba(30, 27, 24, 0.12);
      border-radius: 28px;
      padding: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
    }}
    .section-title {{
      margin: 0 0 16px;
      font-size: 1.8rem;
      line-height: 1.04;
      letter-spacing: -0.03em;
    }}
    .section-subtitle {{
      margin: -8px 0 18px;
      color: var(--muted);
      max-width: 78ch;
    }}
    .score-stack {{
      display: grid;
      gap: 18px;
    }}
    .overall-card {{
      background: linear-gradient(135deg, #fff8ef, #f5e8d4);
      border: 2px solid rgba(30, 27, 24, 0.18);
      border-radius: 26px;
      padding: 24px;
      display: grid;
      gap: 18px;
      grid-template-columns: 0.9fr 1.4fr;
      align-items: end;
    }}
    .overall-score {{
      display: flex;
      align-items: baseline;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .overall-score strong {{
      font-size: clamp(3rem, 7vw, 6rem);
      line-height: 0.9;
      letter-spacing: -0.06em;
    }}
    .overall-score span {{
      font-size: 1rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }}
    .summary-layout,
    .summary-left,
    .areas-grid,
    .rubric-grid,
    .insight-grid,
    .policy-grid {{
      display: grid;
      gap: 18px;
    }}
    .summary-layout {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
      align-items: stretch;
    }}
    .summary-left {{
      grid-template-rows: auto auto;
    }}
    .areas-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .rubric-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .insight-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .policy-grid {{
      grid-template-columns: 1fr;
    }}
    .readout-card,
    .rubric-card,
    .insight-card,
    .policy-card,
    .field-card {{
      background: var(--card);
      border: 2px solid rgba(30, 27, 24, 0.14);
      border-radius: 24px;
      padding: 18px;
    }}
    .readout-card {{
      min-height: 168px;
      display: grid;
      align-content: start;
      gap: 10px;
    }}
    .readout-card.tall {{
      min-height: 100%;
      align-content: start;
    }}
    .rubric-card {{
      display: grid;
      gap: 18px;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 0.8rem;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .eyebrow::before {{
      content: "";
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: linear-gradient(135deg, #1f8f6b, #f0c23f);
      border: 1px solid rgba(0, 0, 0, 0.16);
    }}
    .readout-card p,
    .rubric-card p,
    .insight-card p,
    .policy-card p,
    .field-card p {{
      margin: 0;
      color: var(--muted);
    }}
    .summary-body {{
      color: var(--muted);
    }}
    .summary-list {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      display: grid;
      gap: 6px;
    }}
    .summary-list li {{
      line-height: 1.35;
    }}
    .speedometer-wrap {{
      display: grid;
      gap: 8px;
      justify-items: center;
    }}
    .speedometer-svg {{
      display: block;
      width: 252px;
      max-width: 100%;
      height: auto;
      filter: drop-shadow(0 10px 18px rgba(24, 20, 16, 0.1));
    }}
    .speedometer-svg.medium {{
      width: 184px;
    }}
    .speedometer-svg.small {{
      width: 100%;
      min-width: 0;
    }}
    .speedometer-scale {{
      width: min(240px, 100%);
      display: flex;
      justify-content: space-between;
      font-size: 0.7rem;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      padding: 0 6px;
    }}
    .speedometer-scale.medium {{
      width: 186px;
      font-size: 0.7rem;
    }}
    .score-line {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .metric-label {{
      margin: 0;
      font-size: 1.35rem;
      letter-spacing: -0.03em;
    }}
    .metric-value {{
      font-size: 2rem;
      font-weight: 800;
      line-height: 1;
    }}
    .subrubric-list {{
      margin-top: 16px;
      overflow-x: auto;
      padding-bottom: 4px;
      scrollbar-width: thin;
    }}
    .subrubric-list.expanded {{
      overflow: visible;
      padding-bottom: 0;
      display: flex;
      justify-content: center;
    }}
    .rubric-layout {{
      display: grid;
      gap: 18px;
      grid-template-columns: minmax(208px, 240px) minmax(0, 1fr);
      align-items: start;
    }}
    .rubric-main {{
      display: grid;
      gap: 14px;
      justify-items: start;
      text-align: left;
    }}
    .rubric-copy {{
      display: grid;
      gap: 10px;
    }}
    .subrubric-grid {{
      display: grid;
      gap: 12px;
      grid-auto-flow: column;
      grid-auto-columns: 108px;
      align-items: stretch;
      justify-content: start;
    }}
    .subrubric-grid.expanded {{
      width: 100%;
      grid-auto-flow: row;
      grid-auto-columns: unset;
      grid-template-columns: repeat(5, minmax(108px, 1fr));
      justify-content: center;
    }}
    .subrubric-row {{
      display: grid;
      gap: 4px;
      grid-template-rows: 34px 1fr;
      align-content: stretch;
      justify-items: center;
      justify-self: stretch;
      width: 108px;
      min-width: 108px;
      min-height: 116px;
      padding: 4px 5px;
      border: 1px solid rgba(30, 27, 24, 0.1);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.58);
    }}
    .rubric-card.risk-alignment {{
      grid-column: 1 / -1;
      justify-self: center;
      width: min(100%, 920px);
    }}
    .rubric-card.risk-alignment .rubric-layout {{
      grid-template-columns: minmax(200px, 228px) minmax(0, 1fr);
      align-items: center;
    }}
    .rubric-card.risk-alignment .rubric-main {{
      align-content: center;
    }}
    .rubric-card.risk-alignment .rubric-copy {{
      justify-items: center;
      text-align: center;
    }}
    .subrubric-head {{
      display: grid;
      gap: 2px;
      text-align: center;
      align-content: start;
      width: 100%;
    }}
    .subrubric-head strong {{
      font-size: 0.76rem;
      line-height: 1.1;
      display: block;
      word-break: break-word;
    }}
    .subrubric-row .speedometer-wrap {{
      align-self: end;
      width: 100%;
    }}
    .subrubric-head span {{
      font-size: 0.68rem;
      color: var(--muted);
      font-weight: 700;
    }}
    .insight-list,
    .evidence-list,
    .score-pill-list {{
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 12px;
    }}
    .insight-item {{
      border: 1px solid rgba(30, 27, 24, 0.12);
      border-radius: 18px;
      padding: 14px;
      background: rgba(255, 255, 255, 0.5);
    }}
    .insight-text {{
      margin: 0;
      font-weight: 700;
      color: var(--ink);
    }}
    .pill-row,
    .tag-row {{
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .pill,
    .tag,
    .doc-trigger {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid rgba(30, 27, 24, 0.12);
      background: #fffaf2;
      font-size: 0.78rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .tag {{
      text-transform: none;
      font-weight: 700;
    }}
    .pill.doc-trigger {{
      font-weight: 800;
    }}
    .doc-trigger {{
      appearance: none;
      cursor: pointer;
      font: inherit;
      text-decoration: none;
    }}
    .doc-trigger:hover {{
      border-color: rgba(15, 118, 110, 0.38);
      color: var(--accent);
      background: #f3fbf8;
    }}
    .doc-trigger:focus-visible {{
      outline: 2px solid rgba(15, 118, 110, 0.45);
      outline-offset: 2px;
    }}
    .doc-trigger.meta {{
      background: rgba(255, 250, 242, 0.08);
      border-color: rgba(255, 250, 242, 0.18);
      color: #f5efe6;
    }}
    .doc-trigger.meta:hover {{
      background: rgba(255, 250, 242, 0.16);
      border-color: rgba(255, 250, 242, 0.34);
      color: #ffffff;
    }}
    .doc-trigger.heading {{
      font-size: 1.25rem;
      padding: 8px 12px;
      text-transform: none;
      letter-spacing: 0;
      color: var(--ink);
    }}
    .doc-trigger.heading:hover {{
      color: var(--accent);
      background: rgba(15, 118, 110, 0.08);
    }}
    .policy-card {{
      padding: 0;
      overflow: hidden;
    }}
    .policy-card > summary {{
      list-style: none;
      cursor: pointer;
      display: grid;
      gap: 16px;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: center;
      padding: 20px;
      background: linear-gradient(145deg, #fffaf1, #f6ecdb);
    }}
    .policy-copy {{
      min-width: 0;
    }}
    .policy-copy p {{
      max-width: none;
    }}
    .policy-score {{
      justify-self: end;
      width: 176px;
      max-width: 100%;
    }}
    .policy-score .speedometer-svg.medium {{
      width: 154px;
    }}
    .policy-card > summary::-webkit-details-marker {{
      display: none;
    }}
    .policy-body {{
      padding: 0 20px 20px;
      display: grid;
      gap: 18px;
    }}
    .policy-stat-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }}
    .policy-stat {{
      background: rgba(255, 250, 242, 0.72);
      border: 1px solid rgba(30, 27, 24, 0.12);
      border-radius: 18px;
      padding: 14px;
    }}
    .policy-stat strong {{
      display: block;
      font-size: 0.78rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .policy-stat span {{
      font-size: 1.6rem;
      font-weight: 800;
    }}
    .field-section {{
      border-top: 1px solid rgba(30, 27, 24, 0.1);
      padding-top: 14px;
    }}
    .field-section summary {{
      cursor: pointer;
      font-size: 1.2rem;
      font-weight: 700;
      color: var(--ink);
    }}
    .field-grid {{
      margin-top: 14px;
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .field-card {{
      display: grid;
      gap: 14px;
    }}
    .field-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
    }}
    .field-head h4 {{
      margin: 0;
      font-size: 1.12rem;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
    }}
    .binary-pill {{
      padding: 5px 9px;
      border-radius: 999px;
      font-size: 0.76rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      border: 1px solid rgba(30, 27, 24, 0.12);
    }}
    .binary-pill.pass {{
      background: rgba(31, 143, 77, 0.12);
      color: #166638;
    }}
    .binary-pill.fail {{
      background: rgba(179, 38, 30, 0.12);
      color: #8a241b;
    }}
    .evidence-block {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 10px;
    }}
    .trace-card {{
      border: 1px solid rgba(30, 27, 24, 0.1);
      background: rgba(255, 255, 255, 0.72);
      border-radius: 16px;
      padding: 12px;
    }}
    .trace-card strong {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      font-size: 0.76rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .trace-card p {{
      color: var(--ink);
      font-size: 0.94rem;
    }}
    details details {{
      border-top: 1px dashed rgba(30, 27, 24, 0.14);
      padding-top: 10px;
      margin-top: 12px;
    }}
    .empty {{
      color: var(--muted);
      font-style: italic;
      border: 1px dashed rgba(30, 27, 24, 0.18);
      border-radius: 16px;
      padding: 14px;
      background: rgba(255, 255, 255, 0.5);
    }}
    .footer-note {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 0.88rem;
    }}
    @media (max-width: 1180px) {{
      .app-shell,
      .app-shell.viewer-open {{
        grid-template-areas:
          "panel"
          "main";
        grid-template-columns: 1fr;
      }}
      .pdf-panel {{
        position: relative;
        height: min(70vh, 720px);
        border-right: 0;
        border-bottom: 1px solid rgba(30, 27, 24, 0.12);
        transform: translateY(-14px);
      }}
      .pdf-resizer {{
        display: none;
      }}
      .app-shell.viewer-open .pdf-panel {{
        transform: translateY(0);
      }}
      .hero,
      .overall-card,
      .policy-card > summary {{
        grid-template-columns: 1fr;
      }}
      .rubric-grid,
      .insight-grid,
      .field-grid,
      .evidence-block,
      .summary-layout,
      .summary-left,
      .areas-grid,
      .policy-stat-grid,
      .meta-grid {{
        grid-template-columns: 1fr;
      }}
      .rubric-layout {{
        grid-template-columns: 1fr;
      }}
      .rubric-card.risk-alignment .subrubric-list.expanded {{
        overflow-x: auto;
        justify-content: start;
        padding-bottom: 4px;
      }}
      .rubric-card.risk-alignment .subrubric-grid.expanded {{
        width: max-content;
        grid-auto-flow: column;
        grid-auto-columns: 108px;
        grid-template-columns: none;
      }}
    }}
  </style>
</head>
<body>
  <div class="app-shell" id="app-shell">
    <aside class="pdf-panel" id="pdf-panel" aria-hidden="true">
      <div class="pdf-resizer" id="pdf-resizer" aria-hidden="true"></div>
      <div class="pdf-panel-inner">
        <div class="pdf-panel-head">
          <div class="pdf-panel-copy">
            <div class="eyebrow">Source PDF</div>
            <h2 id="pdf-title">Select a document tag</h2>
            <p id="pdf-subtitle">Click any `policy#` or `research#` tag to open the paper here.</p>
          </div>
          <div class="pdf-panel-actions">
            <a class="pdf-action" id="pdf-open-new" href="#" target="_blank" rel="noopener noreferrer">Open Tab</a>
            <button class="pdf-close" id="pdf-close" type="button">Close</button>
          </div>
        </div>
        <div class="pdf-panel-status" id="pdf-status">Waiting for a document selection.</div>
        <div class="pdf-viewer" id="pdf-viewer">
          <div class="pdf-stage" id="pdf-stage">
            <div class="pdf-page-wrap" id="pdf-page-wrap" hidden>
            </div>
          </div>
        </div>
      </div>
    </aside>
    <main class="shell">
    <section class="hero">
      <div>
        <div class="eyebrow">Combined Research Dashboard</div>
        <h1>{escape(title)}</h1>
      </div>
      <div class="meta-grid">
        <div class="meta-chip">
          <strong>Research</strong>
          <span>{research_tag}</span>
        </div>
        <div class="meta-chip">
          <strong>Policies Compared</strong>
          <span>{len(policy_ids)}</span>
        </div>
        <div class="meta-chip">
          <strong>Generated</strong>
          <span>{escape(generated_at)}</span>
        </div>
        <div class="meta-chip" style="grid-column: 1 / -1;">
          <strong>Policy Set</strong>
          <span class="tag-row">{policy_set_tags}</span>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="score-stack">
        <div class="summary-layout">
          {executive_summary}
        </div>
        <div class="rubric-grid">
          {"".join(rubric_cards)}
        </div>
      </div>
    </section>

    <section class="section">
      <h2 class="section-title">Cross-Policy Insights</h2>
      <p class="section-subtitle">These repeated insights are merged across all policies, so you can quickly see what consistently aligns, what consistently misses, and which bridge actions come up most often.</p>
      <div class="insight-grid">
        {self._render_insight_category("Shared Features", "Repeated alignment themes between the research and the policy set.", highlights["shared_features"], self.config.ui.max_highlights_per_category)}
        {self._render_insight_category("Policy Requirements Not Covered", "Recurring policy expectations the research still does not satisfy clearly.", highlights["policy_requirements_not_covered"], self.config.ui.max_highlights_per_category)}
        {self._render_insight_category("Research Capabilities Not Used", "Research strengths that policy documents are not clearly taking advantage of.", highlights["research_capabilities_not_used"], self.config.ui.max_highlights_per_category)}
        {self._render_insight_category("Bridge Actions", "Practical moves that would make the research more operationally useful to policy actors.", highlights["bridge_actions"], self.config.ui.max_highlights_per_category)}
      </div>
    </section>

    <section class="section">
      <h2 class="section-title">Policy Drill-Down</h2>
      <p class="section-subtitle">The dashboard above is combined per research paper. The panels below keep the policy-by-policy field results, rationales, and chunk-level evidence for anyone who wants to trace an insight back to the source text.</p>
      <div class="policy-grid">
        {"".join(policy_panels)}
      </div>
      <p class="footer-note">All evidence traces come from extraction checkpoints and preserve the original document id, section, field, chunk id, page, and exact text span.</p>
    </section>
    </main>
  </div>
  <script type="module">
    import * as pdfjsLib from "https://cdn.jsdelivr.net/npm/pdfjs-dist@5.7.284/build/pdf.min.mjs";

    pdfjsLib.GlobalWorkerOptions.workerSrc =
      "https://cdn.jsdelivr.net/npm/pdfjs-dist@5.7.284/build/pdf.worker.min.mjs";

    const DOCUMENT_LINKS = {document_links_json};
    const CHUNK_TEXTS = {chunk_text_map_json};
    (() => {{
      const appShell = document.getElementById("app-shell");
      const pdfPanel = document.getElementById("pdf-panel");
      const pdfViewer = document.getElementById("pdf-viewer");
      const pdfStage = document.getElementById("pdf-stage");
      const pdfTitle = document.getElementById("pdf-title");
      const pdfSubtitle = document.getElementById("pdf-subtitle");
      const pdfStatus = document.getElementById("pdf-status");
      const pdfOpenNew = document.getElementById("pdf-open-new");
      const pdfClose = document.getElementById("pdf-close");
      const pdfResizer = document.getElementById("pdf-resizer");
      const mobileQuery = window.matchMedia("(max-width: 1180px)");
      const minPanelWidth = 320;
      const maxPanelRatio = 0.7;
      const pdfCache = new Map();
      let currentPdf = null;
      let currentDocumentId = null;
      let currentPageNumber = 1;
      let currentEvidenceText = "";
      let renderToken = 0;
      let resizeTimer = null;

      function setPanelWidth(width) {{
        const boundedWidth = Math.min(
          Math.max(width, minPanelWidth),
          Math.floor(window.innerWidth * maxPanelRatio),
        );
        document.documentElement.style.setProperty("--panel-width", `${{boundedWidth}}px`);
      }}

      function resetPanelWidth() {{
        document.documentElement.style.setProperty("--panel-width", "minmax(320px, 30vw)");
      }}

      function buildPdfUrl(documentId) {{
        const record = DOCUMENT_LINKS[documentId];
        if (!record) {{
          return null;
        }}
        return record.href;
      }}

      function normalizeText(value) {{
        return String(value || "")
          .toLowerCase()
          .replace(/[^\p{{L}}\p{{N}}\s]+/gu, " ")
          .replace(/\s+/g, " ")
          .trim();
      }}

      function buildNormalizedPageIndex(items) {{
        let text = "";
        const charToItem = [];

        items.forEach((item, index) => {{
          const normalized = normalizeText(item.str);
          if (!normalized) {{
            return;
          }}
          if (text && !text.endsWith(" ")) {{
            text += " ";
            charToItem.push(index);
          }}
          for (const char of normalized) {{
            text += char;
            charToItem.push(index);
          }}
        }});

        return {{ text, charToItem }};
      }}

      function buildHighlightIndexes(items, evidenceText) {{
        const normalizedEvidence = normalizeText(evidenceText);
        if (!normalizedEvidence) {{
          return new Set();
        }}

        const pageIndex = buildNormalizedPageIndex(items);
        const matchStart = pageIndex.text.indexOf(normalizedEvidence);
        if (matchStart >= 0) {{
          const matchEnd = matchStart + normalizedEvidence.length;
          const matchedItems = new Set();
          for (let i = matchStart; i < matchEnd; i += 1) {{
            const itemIndex = pageIndex.charToItem[i];
            if (itemIndex !== undefined) {{
              matchedItems.add(itemIndex);
            }}
          }}
          if (matchedItems.size) {{
            return matchedItems;
          }}
        }}

        const evidenceTokens = new Set(
          normalizedEvidence.split(" ").filter((token) => token.length >= 4),
        );
        const fallbackMatches = new Set();
        items.forEach((item, index) => {{
          const normalizedItem = normalizeText(item.str);
          if (!normalizedItem) {{
            return;
          }}
          for (const token of evidenceTokens) {{
            if (normalizedItem.includes(token)) {{
              fallbackMatches.add(index);
              break;
            }}
          }}
        }});
        return fallbackMatches;
      }}

      function clearPdfStage() {{
        pdfStage.innerHTML = "";
      }}

      function setPdfError(message) {{
        clearPdfStage();
        pdfStatus.textContent = message;
      }}

      function scrollViewerToTarget(target) {{
        if (!target) {{
          return;
        }}
        let relativeTop = 0;
        let node = target;
        while (node && node !== pdfViewer) {{
          relativeTop += node.offsetTop || 0;
          node = node.offsetParent;
        }}
        const topPadding = 28;
        pdfViewer.scrollTo({{
          top: Math.max(0, relativeTop - topPadding),
          behavior: "smooth",
        }});
      }}

      function getViewportScale(page) {{
        const unscaledViewport = page.getViewport({{ scale: 1 }});
        const availableWidth = Math.max(pdfViewer.clientWidth - 36, 220);
        return availableWidth / unscaledViewport.width;
      }}

      function waitForPanelOpen() {{
        return new Promise((resolve) => {{
          if (mobileQuery.matches || !appShell.classList.contains("viewer-open")) {{
            window.requestAnimationFrame(() => resolve());
            return;
          }}

          let settled = false;
          const finish = () => {{
            if (settled) {{
              return;
            }}
            settled = true;
            pdfPanel.removeEventListener("transitionend", onTransitionEnd);
            resolve();
          }};

          const onTransitionEnd = (event) => {{
            if (event.target === pdfPanel || event.target === appShell) {{
              finish();
            }}
          }};

          pdfPanel.addEventListener("transitionend", onTransitionEnd);
          window.setTimeout(finish, 280);
        }});
      }}

      function waitForViewerWidthStable() {{
        return new Promise((resolve) => {{
          let frameCount = 0;
          let stableFrames = 0;
          let previousWidth = 0;

          const tick = () => {{
            const width = pdfViewer.clientWidth;
            if (Math.abs(width - previousWidth) <= 2) {{
              stableFrames += 1;
            }} else {{
              stableFrames = 0;
            }}
            previousWidth = width;
            frameCount += 1;

            if (stableFrames >= 2 || frameCount >= 12) {{
              resolve();
              return;
            }}
            window.requestAnimationFrame(tick);
          }};

          window.requestAnimationFrame(tick);
        }});
      }}

      async function loadPdf(record) {{
        if (pdfCache.has(record.href)) {{
          return pdfCache.get(record.href);
        }}
        const pdfPromise = (async () => {{
          if (window.location.protocol === "file:") {{
            throw new Error("PDF highlighting needs an HTTP(S) page. Open this dashboard through GitHub Pages or a local web server.");
          }}
          const response = await fetch(record.href, {{ cache: "force-cache" }});
          if (!response.ok) {{
            throw new Error(`PDF request failed (${{response.status}} ${{response.statusText}}) for ${{record.href}}.`);
          }}
          const data = new Uint8Array(await response.arrayBuffer());
          const loadingTask = pdfjsLib.getDocument({{ data }});
          return loadingTask.promise;
        }})().catch((error) => {{
          pdfCache.delete(record.href);
          throw error;
        }});
        pdfCache.set(record.href, pdfPromise);
        return pdfPromise;
      }}

      function renderHighlightLayer(highlightLayer, viewport, textContent, evidenceText) {{
        const matchedIndexes = buildHighlightIndexes(textContent.items, evidenceText);
        highlightLayer.innerHTML = "";
        if (!matchedIndexes.size) {{
          return [];
        }}

        const fragment = document.createDocumentFragment();
        const highlightElements = [];
        textContent.items.forEach((item, index) => {{
          if (!matchedIndexes.has(index)) {{
            return;
          }}
          const tx = pdfjsLib.Util.transform(viewport.transform, item.transform);
          const fontHeight = Math.hypot(tx[2], tx[3]);
          const left = tx[4];
          const top = tx[5] - fontHeight;
          const width = Math.max(item.width * viewport.scale, 6);
          const height = Math.max(fontHeight, 10);
          const box = document.createElement("div");
          box.className = "pdf-highlight";
          box.style.left = `${{left}}px`;
          box.style.top = `${{top}}px`;
          box.style.width = `${{width}}px`;
          box.style.height = `${{height}}px`;
          fragment.appendChild(box);
          highlightElements.push(box);
        }});
        highlightLayer.appendChild(fragment);
        return highlightElements;
      }}

      async function renderCurrentPdf() {{
        if (!currentPdf || !currentDocumentId) {{
          clearPdfStage();
          return;
        }}
        try {{
          const token = ++renderToken;
          pdfStatus.textContent = "Rendering PDF...";
          clearPdfStage();
          const safePageNumber = Math.min(
            Math.max(currentPageNumber, 1),
            currentPdf.numPages,
          );
          let targetPageWrap = null;
          let firstHighlight = null;

          for (let pageNumber = 1; pageNumber <= currentPdf.numPages; pageNumber += 1) {{
            const page = await currentPdf.getPage(pageNumber);
            if (token !== renderToken) {{
              return;
            }}

            const scale = getViewportScale(page);
            const viewport = page.getViewport({{ scale }});
            const outputScale = window.devicePixelRatio || 1;
            const pageWrap = document.createElement("div");
            pageWrap.className = "pdf-page-wrap";
            pageWrap.dataset.pageNumber = String(pageNumber);
            pageWrap.style.width = `${{viewport.width}}px`;

            const canvas = document.createElement("canvas");
            canvas.className = "pdf-canvas";
            canvas.width = Math.floor(viewport.width * outputScale);
            canvas.height = Math.floor(viewport.height * outputScale);
            canvas.style.width = `${{viewport.width}}px`;
            canvas.style.height = `${{viewport.height}}px`;

            const highlightLayer = document.createElement("div");
            highlightLayer.className = "pdf-highlight-layer";
            highlightLayer.style.width = `${{viewport.width}}px`;
            highlightLayer.style.height = `${{viewport.height}}px`;

            pageWrap.appendChild(canvas);
            pageWrap.appendChild(highlightLayer);
            pdfStage.appendChild(pageWrap);

            const canvasContext = canvas.getContext("2d");
            canvasContext.setTransform(outputScale, 0, 0, outputScale, 0, 0);
            await page.render({{
              canvasContext,
              viewport,
            }}).promise;
            if (token !== renderToken) {{
              return;
            }}

            const textContent = await page.getTextContent();
            if (token !== renderToken) {{
              return;
            }}
            const highlights = renderHighlightLayer(
              highlightLayer,
              viewport,
              textContent,
              pageNumber === safePageNumber ? currentEvidenceText : "",
            );

            if (pageNumber === safePageNumber) {{
              targetPageWrap = pageWrap;
              if (highlights.length) {{
                firstHighlight = highlights[0];
              }}
            }}
          }}

          pdfStatus.textContent = "";
          const scrollTarget = firstHighlight || targetPageWrap;
          scrollViewerToTarget(scrollTarget);
        }} catch (error) {{
          const message = error instanceof Error ? error.message : String(error);
          setPdfError(`PDF render failed: ${{message}}`);
        }}
      }}

      function resolveHighlightText(documentId, chunkId, evidenceText) {{
        if (chunkId && CHUNK_TEXTS[documentId] && CHUNK_TEXTS[documentId][chunkId]) {{
          return CHUNK_TEXTS[documentId][chunkId];
        }}
        return evidenceText || "";
      }}

      async function openDocument(documentId, page, evidenceText = "", chunkId = "") {{
        const record = DOCUMENT_LINKS[documentId];
        const url = buildPdfUrl(documentId);
        if (!record || !url) {{
          pdfStatus.textContent = `No PDF path is configured for ${{documentId}}.`;
          return;
        }}
        appShell.classList.add("viewer-open");
        pdfPanel.setAttribute("aria-hidden", "false");
        pdfTitle.textContent = record.label;
        pdfSubtitle.textContent = page === "" || page === undefined
          ? `Viewing ${{record.label}}`
          : `Viewing ${{record.label}} starting near page ${{Number(page) + 1}}`;
        pdfStatus.textContent = "Loading PDF...";
        pdfOpenNew.href = page === "" || page === undefined
          ? url
          : `${{url}}#page=${{Number(page) + 1}}`;
        currentDocumentId = documentId;
        currentPageNumber = Number.isFinite(Number(page)) && Number(page) >= 0 ? Number(page) + 1 : 1;
        currentEvidenceText = resolveHighlightText(documentId, chunkId, evidenceText);
        try {{
          await waitForPanelOpen();
          await waitForViewerWidthStable();
          currentPdf = await loadPdf(record);
          await renderCurrentPdf();
        }} catch (error) {{
          currentPdf = null;
          const message = error instanceof Error ? error.message : String(error);
          setPdfError(`PDF load failed: ${{message}}`);
        }}
      }}

      document.addEventListener("click", (event) => {{
        const trigger = event.target.closest("[data-document-id]");
        if (!trigger) {{
          return;
        }}
        event.preventDefault();
        openDocument(
          trigger.dataset.documentId,
          trigger.dataset.page,
          trigger.dataset.evidenceText || "",
          trigger.dataset.chunkId || "",
        );
      }});

      pdfClose.addEventListener("click", () => {{
        appShell.classList.remove("viewer-open");
        pdfPanel.setAttribute("aria-hidden", "true");
        currentPdf = null;
        currentDocumentId = null;
        currentPageNumber = 1;
        currentEvidenceText = "";
        clearPdfStage();
        pdfTitle.textContent = "Select a document tag";
        pdfSubtitle.textContent = "Click any `policy#` or `research#` tag to open the paper here.";
        pdfStatus.textContent = "Waiting for a document selection.";
        pdfOpenNew.href = "#";
      }});

      pdfResizer.addEventListener("pointerdown", (event) => {{
        if (mobileQuery.matches || !appShell.classList.contains("viewer-open")) {{
          return;
        }}
        event.preventDefault();
        const startX = event.clientX;
        const startWidth = pdfPanel.getBoundingClientRect().width;
        pdfResizer.classList.add("dragging");
        pdfResizer.setPointerCapture(event.pointerId);

        const onMove = (moveEvent) => {{
          const delta = startX - moveEvent.clientX;
          setPanelWidth(startWidth + delta);
        }};

        const onUp = () => {{
          pdfResizer.classList.remove("dragging");
          window.removeEventListener("pointermove", onMove);
          window.removeEventListener("pointerup", onUp);
          window.removeEventListener("pointercancel", onUp);
          if (currentPdf) {{
            renderCurrentPdf();
          }}
        }};

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
        window.addEventListener("pointercancel", onUp);
      }});

      window.addEventListener("resize", () => {{
        clearTimeout(resizeTimer);
        if (mobileQuery.matches) {{
          resetPanelWidth();
          return;
        }}
        const currentWidth = pdfPanel.getBoundingClientRect().width;
        if (appShell.classList.contains("viewer-open") && currentWidth > 0) {{
          setPanelWidth(currentWidth);
        }}
        resizeTimer = window.setTimeout(() => {{
          if (currentPdf && appShell.classList.contains("viewer-open")) {{
            renderCurrentPdf();
          }}
        }}, 120);
      }});
    }})();
  </script>
</body>
</html>
"""

    def _render_executive_summary(
        self,
        *,
        overall_score: int,
        policy_count: int,
        strongest: list[tuple[str, int]],
        weakest: list[tuple[str, int]],
        bridge_actions: list[dict[str, Any]],
    ) -> str:
        best_list = self._render_summary_list(
            [f"{_humanize(name)} ({score})" for name, score in strongest[:3]],
            empty_message="No scored rubrics yet.",
        )
        weak_list = self._render_summary_list(
            [f"{_humanize(name)} ({score})" for name, score in weakest[:3]],
            empty_message="No weak areas identified.",
        )
        bridge_list = self._render_summary_list(
            [str(item.get("text", "")).strip() for item in bridge_actions[:7] if str(item.get("text", "")).strip()],
            empty_message="No bridge actions were generated yet.",
        )
        return f"""
        <div class="summary-left">
          <article class="overall-card">
            <div>
              <div class="eyebrow">Overall Score</div>
              <h3 class="metric-label">Combined Research Fit</h3>
              <p>{policy_count} policy comparison reports contributed to this score.</p>
            </div>
            <div>
              {self._render_speedometer(overall_score)}
            </div>
          </article>
          <div class="areas-grid">
            <article class="readout-card">
              <div class="eyebrow">Strongest Areas</div>
              <div class="summary-body">{best_list}</div>
            </article>
            <article class="readout-card">
              <div class="eyebrow">Watch Areas</div>
              <div class="summary-body">{weak_list}</div>
            </article>
          </div>
        </div>
        <article class="readout-card tall">
          <div class="eyebrow">Best Next Move</div>
          <div class="summary-body">{bridge_list}</div>
        </article>
        """

    def _render_summary_list(self, items: list[str], *, empty_message: str) -> str:
        cleaned = [_normalize_space(item) for item in items if _normalize_space(item)]
        if not cleaned:
            return escape(empty_message)
        entries = "".join(f"<li>{escape(item)}</li>" for item in cleaned)
        return f'<ul class="summary-list">{entries}</ul>'

    def _render_rubric_card(
        self,
        rubric: Any,
        rubric_detail: dict[str, Any],
        subrubric_detail_map: dict[str, dict[str, Any]],
    ) -> str:
        score = int(rubric_detail.get("score", 0))
        total_subrubrics = len(rubric.subrubrics)
        rubric_slug = rubric.name.replace("_", "-")
        is_risk = rubric.name == "risk_alignment"
        rubric_classes = f"rubric-card {rubric_slug}"
        subrubric_list_class = "subrubric-list expanded" if is_risk else "subrubric-list"
        subrubric_grid_class = (
            f"subrubric-grid expanded count-{total_subrubrics}"
            if is_risk
            else f"subrubric-grid count-{total_subrubrics}"
        )
        subrubric_rows = [
            self._render_subrubric_row(
                subrubric,
                subrubric_detail_map.get(subrubric.name, {}),
                position=index + 1,
                total=total_subrubrics,
            )
            for index, subrubric in enumerate(rubric.subrubrics)
        ]
        return f"""
        <article class="{rubric_classes}">
          <div class="rubric-layout">
            <div class="rubric-main">
              <div class="eyebrow">Rubric</div>
              <h3 class="metric-label">{escape(_humanize(rubric.name))}</h3>
              {self._render_speedometer(score, size="medium")}
            </div>
            <div class="rubric-copy">
              <p>{escape(rubric.description)}</p>
              <div class="{subrubric_list_class}">
                <div class="{subrubric_grid_class}">
                  {"".join(subrubric_rows)}
                </div>
              </div>
            </div>
          </div>
        </article>
        """

    def _render_subrubric_row(
        self,
        subrubric: Any,
        detail: dict[str, Any],
        *,
        position: int,
        total: int,
    ) -> str:
        del total
        score = int(detail.get("score", 0))
        return f"""
        <div class="subrubric-row pos-{position}">
          <div class="subrubric-head">
            <strong>{escape(_humanize(subrubric.name))}</strong>
          </div>
          {self._render_speedometer(score, size="small", show_scale=False)}
        </div>
        """

    def _render_speedometer(
        self,
        score: int,
        *,
        size: str = "large",
        show_scale: bool = False,
    ) -> str:
        clamped = max(0, min(score, 100))
        is_small = size == "small"
        angle = 180.0 - (clamped * 1.8)
        outer_radius = 98.0 if is_small else 92.0
        needle_start_radius = 48.0 if is_small else 54.0
        needle_end_radius = 95.0 if is_small else 88.0
        face_radius = 58.0 if is_small else 60.0
        ring_radius = 66.0 if is_small else 60.0
        needle_start = _polar_point(120.0, 120.0, needle_start_radius, angle)
        needle_end = _polar_point(120.0, 120.0, needle_end_radius, angle)
        segment_paths = [
            ("#eb3324", _arc_path(120.0, 120.0, outer_radius, 180.0, 142.0)),
            ("#ff7a00", _arc_path(120.0, 120.0, outer_radius, 142.0, 110.0)),
            ("#ffd400", _arc_path(120.0, 120.0, outer_radius, 110.0, 78.0)),
            ("#7ce000", _arc_path(120.0, 120.0, outer_radius, 78.0, 0.0)),
        ]
        scale_class = "medium" if size == "medium" else ""
        scale_html = (
            f"""
            <div class="speedometer-scale {scale_class}">
              <span>Low</span>
              <span>Mid</span>
              <span>High</span>
            </div>
            """
            if show_scale
            else ""
        )
        caption = ""
        score_font = "44" if size == "large" else "38" if size == "medium" else "31"
        score_y = "106" if size == "large" else "104" if size == "medium" else "104"
        caption_y = "114" if size == "large" else "112" if size == "medium" else ""
        caption_svg = (
            f'<text x="120" y="{caption_y}" text-anchor="middle" font-size="9" font-weight="800" letter-spacing="0.08em" fill="#7b756e">{escape(caption.upper())}</text>'
            if caption
            else ""
        )
        svg_class = "speedometer-svg medium" if size == "medium" else "speedometer-svg small" if size == "small" else "speedometer-svg"
        ring_fill = "#f3ede4" if size == "large" else "#f6f1e7"
        needle_stroke = "6" if is_small else "6"
        hub_radius = "4.2" if is_small else "4.5"
        hub_stroke = "1.3" if is_small else "1.5"
        return f"""
        <div class="speedometer-wrap">
          <svg class="{svg_class}" viewBox="0 0 240 138" role="img" aria-label="Score {clamped} out of 100">
            <path d="M 16 120 A 104 104 0 0 1 224 120" fill="none" stroke="#e7dfd3" stroke-width="30" stroke-linecap="butt"></path>
            {"".join(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="24" stroke-linecap="butt"></path>'
                for color, path in segment_paths
            )}
            <circle cx="120" cy="120" r="70" fill="{ring_fill}"></circle>
            <circle cx="120" cy="120" r="{face_radius}" fill="#ffffff"></circle>
            <path d="M 12 120 L 228 120" stroke="#d9d1c5" stroke-width="6" stroke-linecap="round"></path>
            <path d="M {120 - ring_radius:.0f} 120 A {ring_radius:.0f} {ring_radius:.0f} 0 0 1 {120 + ring_radius:.0f} 120" fill="none" stroke="#b9b3ac" stroke-width="5" stroke-linecap="round"></path>
            <line x1="{needle_start[0]:.2f}" y1="{needle_start[1]:.2f}" x2="{needle_end[0]:.2f}" y2="{needle_end[1]:.2f}" stroke="#3a3a3a" stroke-width="{needle_stroke}" stroke-linecap="round"></line>
            <circle cx="{needle_start[0]:.2f}" cy="{needle_start[1]:.2f}" r="{hub_radius}" fill="#575757" stroke="#ffffff" stroke-width="{hub_stroke}"></circle>
            <text x="120" y="{score_y}" text-anchor="middle" font-size="{score_font}" font-weight="900" fill="#454545">{clamped}</text>
            {caption_svg}
          </svg>
          {scale_html}
        </div>
        """

    def _render_insight_category(
        self,
        title: str,
        description: str,
        insights: list[dict[str, Any]],
        max_items: int,
    ) -> str:
        body = self._render_insight_list(insights[:max_items])
        if not body:
            body = '<div class="empty">No insights were available for this category yet.</div>'
        return f"""
        <article class="insight-card">
          <div class="eyebrow">{escape(title)}</div>
          <h3>{escape(title)}</h3>
          <p>{escape(description)}</p>
          <div style="margin-top: 16px;">
            {body}
          </div>
        </article>
        """

    def _render_insight_list(self, insights: list[dict[str, Any]]) -> str:
        if not insights:
            return ""
        items = [self._render_insight_item(insight) for insight in insights]
        return f'<ul class="insight-list">{"".join(items)}</ul>'

    def _render_insight_item(self, insight: dict[str, Any]) -> str:
        policies = insight.get("policy_ids", [])
        tags = "".join(self._render_document_tag(policy_id, variant="pill") for policy_id in policies)
        evidence_details = self._render_evidence_details(
            research_evidence=insight.get("research_evidence", []),
            policy_evidence=insight.get("policy_evidence", []),
        )
        return f"""
        <li class="insight-item">
          <p class="insight-text">{escape(insight.get("text", ""))}</p>
          <div class="pill-row">
            {tags}
          </div>
          {evidence_details}
        </li>
        """

    def _render_policy_panel(self, report: dict[str, Any]) -> str:
        policy_id = next(iter(report.get("policy_document_ids", [])), "policy")
        field_results = report.get("field_results", [])
        grouped_fields = self._group_field_results(field_results)
        section_blocks = []
        for section_name, section_items in grouped_fields.items():
            cards = "".join(self._render_field_card(item) for item in section_items)
            section_blocks.append(
                f"""
                <details class="field-section">
                  <summary>{escape(_humanize(section_name))} · {len(section_items)} field(s)</summary>
                  <div class="field-grid">
                    {cards}
                  </div>
                </details>
                """
            )
        return f"""
        <details class="policy-card">
          <summary>
            <div class="policy-copy">
              <div class="eyebrow">Policy Comparison</div>
              <h3>{self._render_document_tag(policy_id, label=_humanize(policy_id), variant="heading")}</h3>
              <p>{escape(report.get("rationale", "No rationale available."))}</p>
            </div>
            <div class="policy-score">
              <div class="metric-label">Pair-Level Fit</div>
              {self._render_speedometer(int(report.get("overall_percent", 0)), size="medium")}
            </div>
          </summary>
          <div class="policy-body">
            <div class="policy-stat-grid">
              <div class="policy-stat"><strong>Field Calls</strong><span>{len(field_results)}</span></div>
              <div class="policy-stat"><strong>Shared Features</strong><span>{len(report.get("shared_features", []))}</span></div>
              <div class="policy-stat"><strong>Policy Gaps</strong><span>{len(report.get("policy_requirements_not_covered", []))}</span></div>
              <div class="policy-stat"><strong>Bridge Actions</strong><span>{len(report.get("bridge_actions", []))}</span></div>
            </div>
            <div class="insight-grid">
              {self._render_insight_category("Shared Features", "Highlights that matched this one policy.", report.get("shared_features", []), self.config.ui.max_policy_insights_per_category)}
              {self._render_insight_category("Policy Requirements Not Covered", "Requirements this policy asked for that the research still misses.", report.get("policy_requirements_not_covered", []), self.config.ui.max_policy_insights_per_category)}
              {self._render_insight_category("Research Capabilities Not Used", "Research strengths this policy does not yet capitalize on.", report.get("research_capabilities_not_used", []), self.config.ui.max_policy_insights_per_category)}
              {self._render_insight_category("Bridge Actions", "Concrete moves that would make this policy fit stronger.", report.get("bridge_actions", []), self.config.ui.max_policy_insights_per_category)}
            </div>
            <div class="field-stack">
              {"".join(section_blocks)}
            </div>
          </div>
        </details>
        """

    def _render_field_card(self, field_result: dict[str, Any]) -> str:
        focus_names = field_result.get("metadata", {}).get("focus_subrubrics", [])
        score_pills = [
            self._render_binary_pill(name, int(field_result.get("subrubric_scores", {}).get(name, 0)))
            for name in focus_names
        ]
        insight_sections = [
            ("Shared Features", field_result.get("shared_features", [])),
            ("Policy Requirements Not Covered", field_result.get("policy_requirements_not_covered", [])),
            ("Research Capabilities Not Used", field_result.get("research_capabilities_not_used", [])),
            ("Bridge Actions", field_result.get("bridge_actions", [])),
        ]
        rendered_insights = []
        for title, insights in insight_sections:
            if not insights:
                continue
            rendered_insights.append(
                f"""
                <details>
                  <summary>{escape(title)} · {len(insights)}</summary>
                  <div style="margin-top: 12px;">
                    {self._render_insight_list(insights)}
                  </div>
                </details>
                """
            )
        return f"""
        <article class="field-card">
          <div class="field-head">
            <h4>{escape(_humanize(field_result.get("field_name", "field")))}</h4>
            <span class="tag">{escape(_humanize(field_result.get("section", "")))}</span>
          </div>
          <p>{escape(field_result.get("rationale", "No rationale available."))}</p>
          <div class="tag-row">
            {"".join(score_pills) if score_pills else '<span class="tag">No focus subrubrics mapped</span>'}
          </div>
          {"".join(rendered_insights) if rendered_insights else '<div class="empty">No field-level insights were generated for this field.</div>'}
        </article>
        """

    def _render_binary_pill(self, name: str, value: int) -> str:
        class_name = "pass" if value else "fail"
        label = "1" if value else "0"
        return f'<span class="binary-pill {class_name}">{escape(_humanize(name))} · {label}</span>'

    def _render_evidence_details(
        self,
        *,
        research_evidence: list[dict[str, Any]],
        policy_evidence: list[dict[str, Any]],
    ) -> str:
        if not research_evidence and not policy_evidence:
            return ""
        research_block = self._render_trace_column("Research evidence", research_evidence)
        policy_block = self._render_trace_column("Policy evidence", policy_evidence)
        return f"""
        <details>
          <summary>Evidence Trail</summary>
          <div class="evidence-block">
            {research_block}
            {policy_block}
          </div>
        </details>
        """

    def _render_trace_column(self, title: str, traces: list[dict[str, Any]]) -> str:
        if not traces:
            return f'<div class="empty">{escape(title)}: none attached.</div>'
        cards = []
        for trace in traces:
            document_id = str(trace.get("document_id", "doc"))
            chunk_id = trace.get("chunk_id") or "chunk"
            page = trace.get("page")
            page_label = f"p.{page}" if page is not None else "page n/a"
            label = f"{trace.get('section', 'section')} · {trace.get('field_name', 'field')}"
            cards.append(
                f"""
                <article class="trace-card">
                  <strong>{self._render_document_tag(document_id, variant="tag", page=page, evidence_text=trace.get("text", ""), chunk_id=trace.get("chunk_id"))} · {escape(label)} · {escape(str(chunk_id))} · {escape(page_label)}</strong>
                  <p>{escape(trace.get("text", ""))}</p>
                </article>
                """
            )
        return f'<div><div class="eyebrow">{escape(title)}</div><div class="evidence-list">{"".join(cards)}</div></div>'

    def _build_document_links(
        self,
        *,
        document_ids: list[str],
        output_path: Path,
    ) -> dict[str, dict[str, str]]:
        links: dict[str, dict[str, str]] = {}
        for document_id in document_ids:
            href = self._document_href(document_id, output_path)
            if not href:
                continue
            links[document_id] = {
                "href": href,
                "label": _humanize(document_id),
            }
        return links

    def _document_href(self, document_id: str, output_path: Path) -> str | None:
        repo_root = Path.cwd().resolve()
        root_dir = (repo_root / self.config.output.document_root_dir).resolve()
        lowered = document_id.lower()
        if lowered.startswith("policy"):
            target = root_dir / "policy" / f"{document_id}.pdf"
        elif lowered.startswith("research"):
            target = root_dir / "research" / f"{document_id}.pdf"
        else:
            target = root_dir / f"{document_id}.pdf"
        output_parent = (repo_root / output_path).resolve().parent if not output_path.is_absolute() else output_path.resolve().parent
        relative = os.path.relpath(target, output_parent)
        return Path(relative).as_posix()

    def _build_chunk_text_map(self, document_ids: list[str]) -> dict[str, dict[str, str]]:
        chunk_map: dict[str, dict[str, str]] = {}
        base_dir = Path("checkpoints/extraction/pass1")
        for document_id in document_ids:
            doc_dir = base_dir / document_id
            if not doc_dir.exists():
                continue
            entries: dict[str, str] = {}
            for path in sorted(doc_dir.glob("*_classify_c*_request.txt")):
                chunk_id = _extract_chunk_id_from_name(path.name)
                if not chunk_id or chunk_id in entries:
                    continue
                chunk_text = _extract_chunk_text_from_request(path)
                if chunk_text:
                    entries[chunk_id] = chunk_text
            if entries:
                chunk_map[document_id] = entries
        return chunk_map

    def _render_document_tag(
        self,
        document_id: str,
        *,
        label: str | None = None,
        variant: str = "tag",
        page: Any | None = None,
        evidence_text: str | None = None,
        chunk_id: str | None = None,
    ) -> str:
        classes = f"doc-trigger {variant}".strip()
        page_attr = f' data-page="{escape(str(page))}"' if page is not None else ""
        evidence_attr = (
            f' data-evidence-text="{escape(evidence_text)}"'
            if evidence_text
            else ""
        )
        chunk_attr = f' data-chunk-id="{escape(chunk_id)}"' if chunk_id else ""
        text = label or document_id
        return (
            f'<button class="{classes}" type="button" data-document-id="{escape(document_id)}"{page_attr}{evidence_attr}{chunk_attr}>'
            f"{escape(text)}"
            "</button>"
        )

    def _aggregate_insights(self, reports: list[dict[str, Any]], field_name: str) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for report in reports:
            policy_id = next(iter(report.get("policy_document_ids", [])), "policy")
            for insight in report.get(field_name, []):
                text = _normalize_space(str(insight.get("text", "")))
                if not text:
                    continue
                key = text.lower()
                entry = merged.setdefault(
                    key,
                    {
                        "text": text,
                        "policy_ids": [],
                        "research_evidence": [],
                        "policy_evidence": [],
                    },
                )
                if policy_id not in entry["policy_ids"]:
                    entry["policy_ids"].append(policy_id)
                entry["research_evidence"] = _merge_traces(
                    entry["research_evidence"],
                    insight.get("research_evidence", []),
                )
                entry["policy_evidence"] = _merge_traces(
                    entry["policy_evidence"],
                    insight.get("policy_evidence", []),
                )
        return sorted(
            merged.values(),
            key=lambda item: (-len(item["policy_ids"]), item["text"].lower()),
        )

    def _group_field_results(self, field_results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in field_results:
            grouped.setdefault(str(item.get("section", "section")), []).append(item)
        return {
            section_name: sorted(
                items,
                key=lambda item: str(item.get("field_name", "")),
            )
            for section_name, items in sorted(grouped.items())
        }

    def _score_color(self, score: int) -> str:
        if score < 25:
            return "#972b1e"
        if score < 50:
            return "#bf5f09"
        if score < 75:
            return "#9a860d"
        return "#1f7f48"


def _merge_traces(existing: list[dict[str, Any]], new_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {
        (
            item.get("document_id"),
            item.get("section"),
            item.get("field_name"),
            item.get("chunk_id"),
            item.get("page"),
            _normalize_space(str(item.get("text", ""))),
        )
        for item in existing
    }
    merged = list(existing)
    for item in new_items:
        signature = (
            item.get("document_id"),
            item.get("section"),
            item.get("field_name"),
            item.get("chunk_id"),
            item.get("page"),
            _normalize_space(str(item.get("text", ""))),
        )
        if signature in seen:
            continue
        merged.append(item)
        seen.add(signature)
    return merged


def _polar_point(cx: float, cy: float, radius: float, angle_deg: float) -> tuple[float, float]:
    radians = math.radians(angle_deg)
    return (
        cx + (radius * math.cos(radians)),
        cy - (radius * math.sin(radians)),
    )


def _arc_path(cx: float, cy: float, radius: float, start_angle: float, end_angle: float) -> str:
    start_x, start_y = _polar_point(cx, cy, radius, start_angle)
    end_x, end_y = _polar_point(cx, cy, radius, end_angle)
    return f"M {start_x:.2f} {start_y:.2f} A {radius:.2f} {radius:.2f} 0 0 1 {end_x:.2f} {end_y:.2f}"


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")


def _humanize(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().title()


def _normalize_space(value: str) -> str:
    return " ".join(value.split())


def _extract_chunk_id_from_name(filename: str) -> str | None:
    match = re.search(r"_classify_(c\d+)_", filename)
    return match.group(1) if match else None


def _extract_chunk_text_from_request(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    marker = "Chunk text:\n"
    start = text.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = text.find("\n\n[request_json]", start)
    if end < 0:
        end = len(text)
    return _normalize_space(text[start:end])
