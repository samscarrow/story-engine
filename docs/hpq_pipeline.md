# HPQ (High-Performance Quality) Pipeline

The HPQ pipeline delivers higher narrative quality with predictable latency by combining:

- Fast-stage planning and candidate generation (7–14B class models)
- Automated quality checks and reranking
- Selective escalation to a 24B+ model for final polish

## Key Concepts

- Fast vs HQ models: choose via ai-lb `/v1/models` or override with `LM_FAST_MODEL` / `LM_HQ_MODEL`.
- Quality gating: escalate when average score < threshold (default 7.5/10) or when `HPQ_FORCE_24B=1`.
- Canary rollout: `--canary 0.1` (or `HPQ_CANARY_PCT`) randomly escalates 10% for evaluation.
- Budgeting: optional per-call time budget `--budget-ms` threads through to providers.

## Runtime Configuration

- LM_ENDPOINT: base endpoint (default from `config.yaml` → ai-lb `http://localhost:8000`)
- LM_FAST_MODEL: fast model id (optional)
- LM_HQ_MODEL: HQ model id (optional)
- HPQ_FORCE_24B: `1` to force HQ finalization for all requests
- HPQ_CACHE_TTL: seconds for in-process cache (default 1800)
- HPQ_STRUCTURED_SCORING: `1` to enable two-pass JSON scoring (freeform → structured)
- LOG_FORMAT: `json` (default) or `text` ; LOG_LEVEL: `INFO` (default)

### config.yaml (preferred)

Add an `hpq:` section to tune defaults project-wide:

```
hpq:
  enabled: false
  canary_pct: 0.0
  structured_scoring: false
  concurrency: 2
  threshold_low: 7.5
  threshold_high: 8.3
```

## CLI Demo

```
uv run story-engine-hpq --title "The Trial" \
  --premise "Roman prefect under pressure decides a prophet's fate" \
  --characters "Pontius Pilate" Caiaphas "Crowd Representative" \
  --candidates 3 --threshold 7.5 --canary 0.0
```

## Integration Notes

- The orchestrator reads providers from `config.yaml` (active provider points to ai-lb). No change needed.
- ai-lb Monitor should discover LM Studio at `sams-macbook-pro:1234`; adjust `SCAN_HOSTS` if needed.
- Metrics are emitted via JSON logs: `hpq.candidate_ms`, `hpq.evaluate_ms`, `hpq.finalize_ms`, `hpq.best_avg`, `hpq.escalate`.

### Capacity Caps (HQ Node)

If the HQ model is heavy, cap concurrency via ai-lb monitor env:

```
DEFAULT_MAXCONN=0
MAXCONN_MAP="host.docker.internal:1234=2,sams-macbook-pro:1234=2"
```

### HQ Model Validation

HPQ probes whether an HQ model id can handle `/v1/chat/completions` and caches the result; it falls back through a list of large model ids (70B→32B→27B→24B→8x7B).

## A/B Harness

Run a small golden set with baseline vs HPQ and get a summary JSON:

```
uv run python scripts/hpq_ab.py --count 2 --candidates 2 --threshold-low 7.5 --threshold-high 8.3 --structured-scoring
```

Options:
- `--input` JSONL with objects: `{ "title", "premise", "characters": [] }`
- `--report` write summary JSON to a path

## Performance Tips

- Bounded parallel candidates: set `HPQOptions.concurrency` (default 2) to reduce wall time.
- Hysteresis thresholds: `threshold_avg` (low) triggers escalation; `threshold_high` (high) prefers keeping fast output.
- HQ model validation: the pipeline probes chat-capability and falls back across large model IDs.
