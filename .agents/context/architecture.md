# Project Architecture

## Summary

An agentic AML suspicious activity detection system. The user submits a natural language query. The agent parses intent, builds a dynamic execution plan, invokes only the required tools from the ML cycle, and returns a structured, auditable result. The agent never follows a fixed pipeline. Every major action is a tool. Every tool returns typed data.

---

## Design Principles

1. **The agent orchestrates; tools compute.** No business logic, scoring, SQL, or ML lives in prompt text or the agent loop.
2. **Every major action is a tool.** EDA, data query, feature engineering, model selection, training, evaluation, scoring, explanation, and reporting are all tools. The agent invokes them selectively.
3. **Every tool returns typed data.** Tools return typed Pydantic models, not raw strings or dicts.
4. **Deterministic by default.** Risk scoring, rule detection, thresholds, and routing use deterministic code. LLM is used only for intent parsing and explanation polish.
5. **Provider neutral.** The LLM layer must stay swappable. Provider choice lives in config, never in business code.
6. **Config-driven.** A single `AppConfig` class owns all runtime settings: env vars, YAML files, provider, thresholds, paths, logging.
7. **Structured and auditable logging.** Every tool call logs: `query_id`, `tool_name`, `step_id`, `start_time`, `end_time`, `rows_in`, `rows_out`, `provider`, `model`, `config_version`, `status`, `error_summary`.
8. **Modules are small and names are specific.** No deep nesting. No magic constants. No silent fallback paths.

---

## Directory Structure

```
ml-activity-detector/
│
├── AGENTS.md                          # Root agent context (abstract identity, rules, skills)
├── README.md                          # Quickstart and usage
├── pyproject.toml                     # Python dependencies (uv/pip)
├── .env                               # Secret values (LLM API keys, DB path)
├── .env.example                       # Template for env vars
│
├── configs/
│   ├── app.yaml                       # Provider choice, model IDs, retry settings, data paths
│   ├── logging.yaml                   # Log level, format, output target, dev vs. prod flags
│   └── thresholds.yaml                # Risk bands, rule parameters, ensemble weights, ML percentile cutoff
│
├── data/
│   ├── raw/                           # Source CSVs: accounts.csv, transactions.csv, alerts.csv
│   └── processed/                     # DuckDB file(s), parquet feature cache
│
├── dataset/                           # Original dataset CSVs (IBM AMLSim synthetic)
│   ├── accounts.csv
│   ├── transactions.csv
│   └── alerts.csv
│
├── sql/
│   └── duckdb_schema.sql              # Table definitions for accounts, transactions, alerts, audit_log, feature_cache
│
├── logs/                              # Structured log output (dev)
│
├── scripts/
│   └── load_data.py                   # One-shot CSV → DuckDB ingestion script
│
├── src/
│   │
│   ├── core/                          # Shared foundation — no domain logic
│   │   ├── __init__.py
│   │   ├── config.py                  # AppConfig: loads env + YAML, single source of truth
│   │   ├── logging.py                 # Structured logger setup (loguru), log helpers
│   │   └── types.py                   # Shared enums and primitive type aliases (RiskLevel, Intent, PatternType, etc.)
│   │
│   ├── schemas/                       # All typed domain models (Pydantic) — shared across layers
│   │   ├── __init__.py
│   │   ├── domain.py                  # AccountRecord, TransactionRecord, AlertRecord, FeatureSet
│   │   ├── contracts.py               # QuerySpec, ExecutionPlan, PlanStep, ToolResult, AgentResponse
│   │   ├── audit.py                   # AuditEvent, ToolCallLog, RunMetadata
│   │   └── risk.py                    # RiskAssessment, FlaggedItem, EscalationAction
│   │
│   ├── storage/                       # Data access layer — all DuckDB interaction lives here
│   │   ├── __init__.py
│   │   ├── duckdb.py                  # Connection lifecycle, query execution, connection pool
│   │   └── repositories.py            # AccountRepo, TransactionRepo, AlertRepo, AuditRepo, FeatureRepo
│   │
│   ├── tools/                         # All executable tools — each is a self-contained module
│   │   ├── __init__.py
│   │   ├── registry.py                # Tool registry: maps tool name → callable, validates args schema
│   │   │
│   │   ├── data_query/                # Direct SQL/DuckDB lookups and aggregations
│   │   │   ├── __init__.py
│   │   │   ├── tool.py                # DataQueryTool: executes filtered queries and aggregations
│   │   │   └── aggregator.py          # Group-by, threshold, count logic (no ML needed)
│   │   │
│   │   ├── eda/                       # Exploratory data analysis — selective, not always run
│   │   │   ├── __init__.py
│   │   │   ├── tool.py                # EDATool: profiling, distributions, volume trends, baselines
│   │   │   └── charts.py              # Chart generation (matplotlib): timeline, histogram, heatmap
│   │   │
│   │   ├── features/                  # Feature engineering — computed on demand per pattern
│   │   │   ├── __init__.py
│   │   │   ├── tool.py                # FeatureTool: dispatches to per-family feature functions
│   │   │   ├── volume.py              # Rolling counts, sums, avg_amount, amount_std
│   │   │   ├── threshold.py           # count_near_threshold, round_number_bias
│   │   │   ├── network.py             # fan_in_degree, fan_out_degree, distinct_counterparties (NetworkX)
│   │   │   └── velocity.py            # velocity_zscore (MAD-based), dwell_time_avg_hours, in_out_ratio
│   │   │
│   │   ├── detection/                 # Anomaly and pattern detection — hybrid rule + ML
│   │   │   ├── __init__.py
│   │   │   ├── tool.py                # DetectionTool: routes to rule engine + ML scorer + combiner
│   │   │   ├── rules.py               # Rule detectors per pattern (structuring, smurfing, layering, rapid_cashout, velocity)
│   │   │   ├── ml_scorer.py           # PyOD IForest/LOF wrapper — fit, score, percentile threshold
│   │   │   └── ensemble.py            # OR-gate combiner: merges rule flags + ML score → raw signals
│   │   │
│   │   ├── scoring/                   # Risk classification — deterministic, config-driven
│   │   │   ├── __init__.py
│   │   │   └── tool.py                # ScoringTool: composite score → risk band, confidence, escalation action
│   │   │
│   │   ├── explanation/               # Human-readable flag reasons — grounded, not hallucinated
│   │   │   ├── __init__.py
│   │   │   ├── tool.py                # ExplanationTool: fills templates, optionally polishes with LLM
│   │   │   └── templates.py           # Pattern-specific templates with numeric drift check
│   │   │
│   │   └── reporting/                 # Final structured output assembly
│   │       ├── __init__.py
│   │       └── tool.py                # ReportingTool: assembles AgentResponse from all tool outputs
│   │
│   ├── llm/                           # LLM provider layer — provider-neutral
│   │   ├── __init__.py
│   │   ├── client.py                  # LLMClient: wraps LangChain ChatModel, exposes invoke/structured_output
│   │   ├── providers/
│   │   │   ├── __init__.py
│   │   │   ├── groq.py                # ChatGroq adapter
│   │   │   ├── ollama.py              # ChatOllama adapter
│   │   │   └── openrouter.py          # ChatOpenAI-compatible OpenRouter adapter
│   │   └── prompts/
│   │       ├── intent_parser.py       # System + user prompt for QuerySpec extraction
│   │       └── explanation_polish.py  # System prompt for phrasing polish (numbers must not change)
│   │
│   ├── agent/                         # Orchestration layer — LangGraph-based
│   │   ├── __init__.py
│   │   ├── state.py                   # AgentState: shared LangGraph state across nodes
│   │   ├── nodes.py                   # Node functions: parse_intent, plan, execute_step, explain, report
│   │   ├── graph.py                   # LangGraph StateGraph: wires nodes and conditional edges
│   │   ├── planner.py                 # Two-tier planner: decision table (Tier 1) + LLM fallback (Tier 2)
│   │   └── guardrails.py              # Tool whitelist check, arg schema validation, step cap enforcement
│   │
│   └── api/
│       ├── __init__.py
│       └── main.py                    # FastAPI app: POST /query endpoint, health check
│
├── tests/
│   ├── test_contracts.py              # Pydantic model validation, serialization round-trips
│   ├── test_planner.py                # Golden tests: 3 canonical queries → exact tools_invoked/skipped
│   ├── test_tools/
│   │   ├── test_data_query.py
│   │   ├── test_eda.py
│   │   ├── test_features.py
│   │   ├── test_detection.py
│   │   ├── test_scoring.py
│   │   └── test_explanation.py
│   ├── test_guardrails.py             # Malformed/out-of-whitelist plans must be rejected before execution
│   └── test_end_to_end.py             # Full query → response contract shape checks
│
└── notebooks/
    └── demo.ipynb                     # Interactive walkthrough of the three canonical queries
```

---

## Module Boundaries

| Module | Owns | Does Not Own |
|---|---|---|
| `core` | Config, logging, shared enums | Domain models, tools, agent logic |
| `schemas` | All typed contracts and domain records | Computation, storage, LLM calls |
| `storage` | DuckDB connection and repository methods | Feature logic, ML, agent routing |
| `tools` | All computation — EDA, features, detection, scoring, explanation, reporting | Agent orchestration, LLM calls, storage |
| `llm` | LangChain provider adapters, prompts | Scoring, rules, any deterministic logic |
| `agent` | LangGraph graph, state, planner, guardrails | Tool implementation, storage, LLM adapters |
| `api` | FastAPI routes | Any business logic |

---

## LangGraph Agent Flow

```
[parse_intent] → [plan] → [execute_step (loop)] → [explain] → [report]
                  ↑_____________ (replan on failure, max 1 replan) ___|
```

- `parse_intent`: LLM call → `QuerySpec` (validated, temp=0)
- `plan`: Tier 1 decision table lookup; Tier 2 LLM fallback if no match; guardrails validate before execution begins
- `execute_step`: iterates `ExecutionPlan.steps`, calls registry, captures `ToolResult`, logs `AuditEvent`
- `explain`: `ExplanationTool` fills templates from `ToolResult` data; optional LLM polish with numeric drift check
- `report`: `ReportingTool` assembles `AgentResponse` from all outputs; writes final audit entry

---

## Tool Registry Contract

Every tool registered in `tools/registry.py` must expose:

```python
class ToolSpec(BaseModel):
    name: str                    # Unique identifier used in ExecutionPlan steps
    description: str             # One sentence; used in LLM-fallback context
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    callable: Callable[..., ToolResult]
```

The registry validates every tool call's args against `input_schema` before dispatch. Unknown tool names are rejected. Unknown args are rejected. This is the guardrail boundary.

---

## Risk Scoring Formula

```
composite_score = 100 × clip(
    w_rule    × (distinct_rule_detectors_triggered / total_rule_detectors_run)
  + w_ml      × normalized_ml_anomaly_score   # percentile rank, 0–1
  + w_context × context_multiplier            # KYC risk rating, PEP flag, prior alert count
, 0, 1)
```

All weights (`w_rule`, `w_ml`, `w_context`) and band thresholds live in `configs/thresholds.yaml`.

| Score | Risk Level | Recommended Action |
|---|---|---|
| 0–39 | Low | Monitor |
| 40–69 | Medium | Flag for review |
| 70–100 | High | Report (SAR consideration) |

**Confidence** = `distinct_detectors_agreed / detectors_run` — a separate axis from the score, shown alongside it.

---

## AML Feature Families

| Family | Features | Notes |
|---|---|---|
| Volume | `txn_count_7d`, `txn_count_30d`, `txn_sum_7d`, `txn_sum_30d`, `avg_amount`, `amount_std` | Vectorized rolling windows |
| Threshold-avoidance | `count_near_threshold_30d` (in [0.85, 0.99] × threshold), `round_number_bias` | Structuring-specific |
| Network | `distinct_counterparties_30d`, `fan_in_degree`, `fan_out_degree` | NetworkX on account graph |
| Velocity/deviation | `velocity_zscore` (MAD-based robust z-score), `dwell_time_avg_hours`, `in_out_ratio_30d` | Rapid cash-out and layering |
| Geographic | `distinct_countries_30d` | Cross-jurisdiction layering |

Each feature family is computed only when the execution plan requests that feature set — never all upfront.

---

## Provider Strategy

Provider adapters in `src/llm/providers/` all return a `langchain_core.language_models.BaseChatModel`. The `LLMClient` in `src/llm/client.py` selects the adapter based on `AppConfig.provider`. Business modules only see `LLMClient`, never a provider-specific class.

Supported providers at MVP:

| Provider | Adapter | Notes |
|---|---|---|
| Groq | `ChatGroq` | Fast inference, low latency |
| Ollama | `ChatOllama` | Local, fully offline |
| OpenRouter | `ChatOpenAI` (compatible) | Multi-model routing, fallback |

---

## Data Schema (DuckDB)

Tables already defined in `sql/duckdb_schema.sql`. Additional tables added during build:

```sql
-- Feature cache (per account × window, keyed by query_hash)
CREATE TABLE IF NOT EXISTS feature_cache (
    cache_key VARCHAR PRIMARY KEY,
    account_id BIGINT,
    window_days INTEGER,
    computed_at TIMESTAMP,
    features_json JSON
);

-- Audit log (append-only, never updated or deleted)
CREATE TABLE IF NOT EXISTS audit_log (
    event_id VARCHAR PRIMARY KEY,
    query_id VARCHAR NOT NULL,
    tool_name VARCHAR,
    step_id INTEGER,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    rows_in INTEGER,
    rows_out INTEGER,
    provider VARCHAR,
    model_name VARCHAR,
    config_version VARCHAR,
    status VARCHAR,      -- 'ok' | 'error' | 'skipped'
    error_summary VARCHAR,
    payload_hash VARCHAR
);

-- Risk assessments (persisted flag store)
CREATE TABLE IF NOT EXISTS risk_assessments (
    assessment_id VARCHAR PRIMARY KEY,
    query_id VARCHAR NOT NULL,
    account_id BIGINT,
    pattern_type VARCHAR,
    rule_flags JSON,
    ml_anomaly_score FLOAT,
    composite_score FLOAT,
    confidence FLOAT,
    risk_level VARCHAR,
    escalation_action VARCHAR,
    explanation_text TEXT,
    config_version VARCHAR,
    created_at TIMESTAMP
);
```

---

## Logging Shape (Development)

Every tool call emits a structured log entry via `loguru` with the following fields:

```json
{
  "query_id": "q_20260724_0001",
  "tool_name": "feature_engineering",
  "step_id": 2,
  "started_at": "2026-07-24T10:00:01.123Z",
  "ended_at": "2026-07-24T10:00:01.890Z",
  "duration_ms": 767,
  "rows_in": 4210,
  "rows_out": 4210,
  "provider": "groq",
  "model_name": "llama-3.3-70b-versatile",
  "config_version": "v1.2.0",
  "status": "ok",
  "error_summary": null
}
```

Errors include `error_summary` (safe, no PII) and stack trace. Audit events written to `audit_log` table mirror the same structure.
