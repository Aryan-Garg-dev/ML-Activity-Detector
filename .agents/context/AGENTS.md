# AGENTS.md — ML Activity Detector

This file is the authoritative abstract reference for the project. Every skill, agent, or contributor working on this repository must read this file before taking any action. It covers identity, directory layout, tool contracts, coding rules, skill triggers, and rulesets.

---

## Project Identity

This repository builds an **agentic AML (Anti-Money Laundering) suspicious activity detection system**. The system accepts a natural language query, autonomously decides what analysis to perform, selects and invokes the minimum set of tools needed to answer that query, and returns a structured, explainable, auditable result.

The system is not a fixed pipeline. Every query gets a different execution plan. The same tool set is invoked in different combinations based on parsed intent.

**Domain:** Financial compliance, anti-money laundering, transaction risk detection.
**Dataset:** IBM AMLSim synthetic transactions. Three tables: `accounts`, `transactions`, `alerts`. Already loaded into `activity.duckdb`.
**Languages:** Python 3.11+. Dependencies managed with `uv` (`pyproject.toml`).

---

## Abstract Architecture

```
[Natural Language Query]
        ↓
[Intent Parser] — LLM call → QuerySpec (validated, temp=0)
        ↓
[Planner] — Tier 1: decision table | Tier 2: LLM fallback (guardrail validated)
        ↓
[ExecutionPlan] — written to audit_log BEFORE execution begins
        ↓
[Tool Executor Loop] — calls registry, captures ToolResult, logs AuditEvent per step
        ↓
[Explanation Generator] — template fill → optional LLM polish → numeric drift check
        ↓
[Reporting] — assembles AgentResponse with execution summary, flagged items, charts
        ↓
[Structured AgentResponse] + [Audit Log Entry]
```

Five routing paths:

| Intent | Tools Invoked | What Is Skipped |
|---|---|---|
| `pattern_search` | load, features, detect, score, explain, report | EDA, aggregation |
| `aggregation_query` | load, data_query, explain(light), report | Features, ML detection, EDA |
| `entity_lookup` | entity_lookup → [miss] load, features, detect, score, explain, report | EDA, batch scan |
| `broad_eda` | load, EDA, (optional) features, detect, score, explain, report | Direct aggregation |
| `risk_scoring_batch` | load, features(all), detect, score, explain, report | EDA, aggregation |

---

## Directory Map

```
ml-activity-detector/
├── AGENTS.md                         # This root context file (abstract rules)
├── .agents/context/                  # Detailed context files for every skill/agent
│   ├── AGENTS.md                     # This file
│   ├── architecture.md               # Directory structure, module boundaries, schemas, logging
│   ├── build-plan.md                 # Phased build plan with scope and exit criteria
│   ├── progress-tracker.md           # Living checklist of all deliverables
│   └── project-overview.md           # Problem statement and system requirements
├── configs/
│   ├── app.yaml                      # Provider, model IDs, data paths, retry settings
│   ├── logging.yaml                  # Log level, format, dev mode flag
│   └── thresholds.yaml               # Risk bands, rule params, ensemble weights, ML cutoff
├── data/raw/                         # Source CSVs
├── data/processed/                   # DuckDB files and parquet feature cache
├── dataset/                          # Original IBM AMLSim CSVs
├── sql/duckdb_schema.sql             # Table definitions
├── scripts/load_data.py              # CSV → DuckDB ingestion
├── src/
│   ├── core/                         # Config, logging, enums
│   ├── schemas/                      # All Pydantic models (domain, contracts, audit, risk)
│   ├── storage/                      # DuckDB connection + repositories
│   ├── tools/                        # All executable tools
│   ├── llm/                          # Provider adapters + prompts
│   ├── agent/                        # LangGraph graph, planner, guardrails, state
│   └── api/                          # FastAPI endpoint
├── tests/                            # Unit, golden, guardrail, end-to-end tests
└── notebooks/                        # Demo notebook
```

Full directory tree and per-file descriptions live in [architecture.md](./architecture.md).

---

## Tool Map

Every major action is a tool. The agent invokes tools selectively — it never runs a tool that is not in the execution plan.

| Tool Name | Module | What It Does |
|---|---|---|
| `data_query` | `src/tools/data_query/` | DuckDB-backed filtered lookups and direct aggregations. No ML. Used for aggregation_query intent. |
| `eda` | `src/tools/eda/` | Profiling, distributions, volume trends, baseline stats, chart paths. Selective — only for broad_eda or explicit exploration. |
| `feature_engineering` | `src/tools/features/` | Computes only the requested feature family. Families: volume, threshold, network, velocity. All vectorized. |
| `detection` | `src/tools/detection/` | Hybrid rule engine + PyOD ML scorer (IForest, LOF) + OR-gate ensemble. Returns raw signals and triggered flags. |
| `scoring` | `src/tools/scoring/` | Composite score formula → risk band (low/medium/high) + confidence. Config-driven. Deterministic. |
| `explanation` | `src/tools/explanation/` | Template-filled grounded explanations. LLM polish is optional and guarded by numeric drift check. |
| `reporting` | `src/tools/reporting/` | Assembles `AgentResponse` from all tool outputs. Includes execution summary, flagged items, charts, metrics. |
| `entity_lookup` | `src/tools/data_query/` | Checks `risk_assessments` table for a cached flag. Cache hit → return directly. Cache miss → triggers single-entity flow. |

All tools are registered in `src/tools/registry.py`. Registration requires: `name`, `description`, `input_schema`, `output_schema`, `callable`. The registry validates args against `input_schema` before dispatching. Unknown tool names and unknown args are both rejected.

---

## Domain Records (Schemas)

All data contracts are typed Pydantic models. No raw dicts passed between tools.

| Model | Module | Purpose |
|---|---|---|
| `AccountRecord` | `schemas/domain.py` | Account metadata from `accounts` table |
| `TransactionRecord` | `schemas/domain.py` | Transaction record from `transactions` table |
| `AlertRecord` | `schemas/domain.py` | Alert record from `alerts` table |
| `FeatureSet` | `schemas/domain.py` | Computed feature vector per account × window |
| `QuerySpec` | `schemas/contracts.py` | Parsed intent: intent type, pattern, filters, entities, aggregation spec |
| `ExecutionPlan` | `schemas/contracts.py` | Ordered, parameterized, justified tool step list + skipped tools + skip reasons |
| `PlanStep` | `schemas/contracts.py` | Single step in the execution plan: tool name, args, reason |
| `ToolResult` | `schemas/contracts.py` | Typed output from a single tool call |
| `AgentResponse` | `schemas/contracts.py` | Final structured output: query spec, execution summary, flagged items, charts, metrics |
| `AuditEvent` | `schemas/audit.py` | Per-tool-call log: query_id, tool, step, times, rows, model, provider, status |
| `RunMetadata` | `schemas/audit.py` | Per-query run metadata: query_id, raw query, total duration, config version |
| `RiskAssessment` | `schemas/risk.py` | Per-entity composite score, confidence, risk level, escalation action |
| `FlaggedItem` | `schemas/risk.py` | Flagged entity with score, explanation, triggered signals, supporting evidence |

---

## Config Class Contract

`AppConfig` in `src/core/config.py` is the single entry point for all runtime settings. It is instantiated once and passed into modules. No module reads env vars directly.

Fields it owns:

| Field Group | Examples |
|---|---|
| Provider | `provider` (groq/ollama/openrouter), `model_id`, `fallback_model_id` |
| LLM | `base_url`, `api_key_env_var`, `temperature`, `max_retries`, `timeout_seconds` |
| Data | `db_path`, `raw_data_dir`, `processed_data_dir`, `feature_cache_ttl_hours` |
| Thresholds | `reporting_threshold`, `structuring_window_days`, `structuring_min_txn_count`, `risk_bands`, `w_rule`, `w_ml`, `w_context`, `ml_contamination`, `ml_percentile_cutoff` |
| Logging | `log_level`, `log_to_file`, `log_dir`, `dev_mode` |
| Runtime | `config_version`, `max_plan_steps`, `ml_random_seed` |

---

## Coding Rules

1. **The agent orchestrates; tools compute.** No business logic, scoring, SQL, or ML in prompt text or the agent loop.
2. **Every tool returns a typed Pydantic model.** No raw strings or dicts between components.
3. **No hardcoded provider choice inside business modules.** Provider lives in `AppConfig`.
4. **No magic constants.** All thresholds, weights, paths, and rule parameters live in `configs/thresholds.yaml` and are accessed through `AppConfig`.
5. **No silent fallback paths.** Every path that skips a tool must write a skip reason to the `ExecutionPlan` and to the audit log.
6. **No raw PII to the LLM.** Only pseudonymous IDs and derived feature values. Enforced at the tool result serialization boundary.
7. **Deterministic first.** Risk scoring, rule detection, thresholds, and Tier 1 routing are deterministic Python. LLM is used at exactly two points: intent parsing and explanation polish.
8. **Modules are small; interfaces are narrow.** One responsibility per module. Public interfaces expose only what is needed.
9. **Prefer pure functions for transforms.** Feature engineering functions are pure: same input, same output, no side effects.
10. **Tests before expansion.** Add tests for contracts and planner behavior before expanding scope to the next phase.
11. **Audit log is append-only.** Never update or delete an audit event. No silent failure paths that skip logging.
12. **The `ExecutionPlan` is written to audit before the first tool executes.** Even a mid-execution failure leaves a record of intent.

---

## Logging and Audit Contract

Every tool call must log:

```json
{
  "query_id": "q_20260724_0001",
  "tool_name": "feature_engineering",
  "step_id": 2,
  "started_at": "ISO8601",
  "ended_at": "ISO8601",
  "duration_ms": 767,
  "rows_in": 4210,
  "rows_out": 4210,
  "provider": "groq | ollama | openrouter | null",
  "model_name": "model-id | null",
  "config_version": "v1.2.0",
  "status": "ok | error | skipped",
  "error_summary": "safe summary, no PII | null"
}
```

Errors also log the stack trace (never raw data). Audit events are written to the `audit_log` DuckDB table AND emitted as structured loguru lines.

---

## Provider Strategy

All providers implement `BaseChatModel` (LangChain). The `LLMClient` wraps the active adapter. Adapters live in `src/llm/providers/`. The `AppConfig.provider` field selects the adapter. Business code only imports `LLMClient`.

| Provider | When to Use |
|---|---|
| Groq | Fast inference, low latency, good for production demo |
| Ollama | Fully local, offline, no API key needed, good for dev |
| OpenRouter | Multi-model routing, good for fallback or model comparison |

To add a new provider: implement the adapter in `src/llm/providers/`, add the `ProviderName` enum value, add the selection branch in `LLMClient`, add to `app.yaml` documentation.

---

## AML Pattern Catalog

| Pattern | `pattern_type` value | Core Features Used | Rule Threshold |
|---|---|---|---|
| Structuring | `structuring` | `count_near_threshold_30d`, `txn_sum_7d`, `txn_count_7d` | ≥3 txns in 7d in [0.85, 0.99] × $10k, sum > $25k |
| Smurfing | `smurfing` | `fan_in_degree`, `distinct_counterparties_30d` | ≥5 distinct senders into one account within 48h |
| Layering | `layering` | `dwell_time_avg_hours`, `distinct_countries_30d`, `fan_out_degree` | ≥3 hops within 24h, minimal balance retention |
| Rapid cash-out | `rapid_cashout` | `dwell_time_avg_hours`, `in_out_ratio_30d` | >80% of inflow withdrawn within 24h vs. 3d+ historical avg |
| Velocity | `velocity` | `velocity_zscore`, `txn_count_30d`, `txn_sum_30d` | `|velocity_zscore| > 3` (configurable, MAD-based) |

---

## Risk Scoring

```
composite_score = 100 × clip(
    w_rule    × (distinct_rule_detectors_triggered / total_rule_detectors_run)
  + w_ml      × normalized_ml_anomaly_score
  + w_context × context_multiplier
, 0, 1)
```

| Score | Risk Level | Escalation Action |
|---|---|---|
| 0–39 | Low | Monitor |
| 40–69 | Medium | Flag for review |
| 70–100 | High | Report (SAR consideration) |

**Confidence** = `distinct_detectors_agreed / detectors_run`. Shown alongside risk level, not instead of it.

Weights, bands, and rule thresholds all live in `configs/thresholds.yaml`. Compliance can retune without a code change.

---

## Skill Triggers

| Skill | Trigger | Scope |
|---|---|---|
| `architect` | Design choice is open, or `/develop` says a decision is owed | Project structure, stack, module design, ADRs |
| `develop` | Building a specific phase deliverable | Feature delivery against build plan |
| `debug` | A tool or flow is broken, test fails unexpectedly | Root cause analysis, minimal fix |
| `test` | A tool or phase is newly complete | Contract tests, golden tests, regression |
| `check` | Before merge or demo handoff | Verify behavior against spec, senior code review |
| `sync` | After a phase is stable | Update AGENTS.md, reconcile scope |
| `scope` | Planning the next phase or feature boundary | Feature sequencing and boundaries |
| `scikit-learn` | Model selection, training, evaluation in Phase 2+ | Detection and scoring patterns |
| `exploratory-data-analysis` | EDA tool implementation or dataset quality review | Profiling and quality checks |
| `matplotlib` | Chart implementation in EDA tool | Plot types and output formats |
| `shap` | Explanation tool — feature contribution views | SHAP waterfall/bar plots for flagged entities |
| `od-expert` | Anomaly detection implementation in Phase 2 | IForest, LOF, ensemble, PyOD patterns |

---

## Non-Goals

- Not a production SAR/CTR filing integration. Output is a recommendation, not a filed report.
- Not real-time streaming ingestion. Batch/query-driven analysis over a loaded dataset.
- Not a KYC/onboarding system. Customer risk ratings are inputs, not derived from scratch.
- Not a supervised learning system. No labeled AML ground truth available publicly; detection stays unsupervised + rule-based.
- No graph database (Neo4j). NetworkX is sufficient at IBM AMLSim scale.
- No DBSCAN. IForest (global anomaly shape) + LOF (local density) is the ensemble. A third detector adds hyperparameters without labeled data to validate them.
- No XGBoost or supervised classifiers at MVP. Would require labeled training data.

---

## Build Phase Summary

| Phase | Focus | Key Output |
|---|---|---|
| 1 | Foundation and contracts | `AppConfig`, all Pydantic models, DuckDB layer, logging |
| 2 | Deterministic tools | All 8 tools built, unit-tested, audit-logged |
| 3 | LLM provider layer and agent glue | Provider adapters, LangGraph graph, planner, guardrails |
| 4 | Query-aware workflows and full response | All 5 routing paths, FastAPI endpoint, CLI |
| 5 | Hardening and review | Test suite, logging validation, README, demo notebook |

Full phase details in [build-plan.md](./build-plan.md). Living checklist in [progress-tracker.md](./progress-tracker.md).
