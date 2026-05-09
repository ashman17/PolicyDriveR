# policydriver

Research-to-policy alignment tooling with a configurable PDF-to-JSON extraction layer.

## extraction

The extraction module now supports:

- PDF or text ingestion via PyMuPDF first, then `unstructured` as fallback
- paragraph-aware chunking around `800-1200` tokens with overlap
- multi-pass extraction:
  - pass 1: section classification
  - pass 2: section-specific field filling
  - final code-based consolidation with evidence-weighted confidence
- Ollama-backed structured extraction using `llama3:8b` and optional comparison passes with `mistral-nemo`
- configurable fields via JSON config, with YAML also supported when `PyYAML` is installed
- evidence anchoring and per-field confidence output

Default field config lives at [examples/config/extraction_fields.json](examples/config/extraction_fields.json).

## quickstart

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Print the default extraction config:

```bash
policydriver extract-config
```

Run extraction with the default Ollama setup:

```bash
policydriver extract-sample --model llama3:8b
```

Force the offline heuristic fallback:

```bash
policydriver extract-sample --backend heuristic --model llama3:8b
```

Run a specific sample PDF from `data/research`:

```bash
policydriver extract-sample --file research2.pdf --model llama3:8b
```

Run extraction on any file with explicit flags:

```bash
policydriver extract --file data/research/research1.pdf --model llama3:8b
```

LLM calls are checkpointed by default, so reruns automatically reuse completed calls:

```bash
policydriver extract --file data/research/research1.pdf --model gemma3:4b
```

Pass 1 and pass 2 checkpoints are scoped by document id, for example:

```text
checkpoints/extraction/pass1/research1/...
checkpoints/extraction/pass2/research1/...
```

## alignment

The alignment module compares one research pass3 folder against one or more policy pass3 folders using:

- rubric and output-field YAML config
- one Ollama call per `section + field`
- exact research and policy evidence text in the user prompt
- binary `0/1` scoring for every configured subrubric

Print the default alignment config:

```bash
policydriver align-config
```

Run alignment from extraction pass3 checkpoints:

```bash
policydriver align-checkpoints --research-id research1 --policy-id policy3 --model gemma3:4b
```

If `--policy-id` is omitted, all folders matching `policy*` under `checkpoints/extraction/pass3` are used.

## scoring

The scoring module combines the single-policy alignment reports for one research document and produces:

- normalized subrubric scores out of 100
- rubric scores out of 100
- one overall score out of 100

Run scoring from saved alignment checkpoints:

```bash
policydriver score-checkpoints --research-id research1
```

## viewer

The viewer module renders one combined HTML dashboard per research paper. It uses:

- all single-policy alignment reports for that research
- the normalized scoring report when available
- policy-level field results and chunk-level evidence traces for drill-down

Print the default viewer config:

```bash
policydriver viewer-config
```

Render the dashboard:

```bash
policydriver render-dashboard --research-id research1
```

By default the HTML is written to:

```text
checkpoints/viewer/research1.html
```
