# Progress Tracker

Status legend: `[ ]` not started · `[/]` in progress · `[x]` complete

---

## Phase 1 — Foundation and Contracts

### Config and Runtime
- [x] Create `configs/app.yaml` with provider, model IDs, data paths, retry settings
- [x] Create `configs/logging.yaml` with log level, format, output settings
- [x] Create `configs/thresholds.yaml` with risk bands, rule parameters, ensemble weights
- [x] Implement `src/core/config.py` — `AppConfig` class loading env + all YAML files
- [x] Implement `src/core/logging.py` — loguru setup, structured log helpers, log field contract
- [x] Implement `src/core/types.py` — `IntentType`, `PatternType`, `RiskLevel`, `EscalationAction`, `ToolName`, `ProviderName` enums

### Domain Models
- [x] Implement `src/schemas/domain.py` — `AccountRecord`, `TransactionRecord`, `AlertRecord`, `FeatureSet`
- [x] Implement `src/schemas/contracts.py` — `QuerySpec`, `ExecutionPlan`, `PlanStep`, `ToolResult`, `AgentResponse`
- [x] Implement `src/schemas/audit.py` — `AuditEvent`, `ToolCallLog`, `RunMetadata`
- [x] Implement `src/schemas/risk.py` — `RiskAssessment`, `FlaggedItem`, `EscalationAction`

### Storage Layer
- [x] Update `sql/duckdb_schema.sql` — add `feature_cache`, `audit_log`, `risk_assessments` tables
- [x] Implement `src/storage/duckdb.py` — connection lifecycle, query executor, parameterized query helper
- [x] Implement `src/storage/repositories.py` — `AccountRepo`, `TransactionRepo`, `AlertRepo`, `AuditRepo`, `FeatureRepo`
- [x] Implement `scripts/load_data.py` — CSV → DuckDB ingestion with schema setup

### Verification
- [x] `AppConfig` loads from env + YAML without error
- [x] DuckDB connects and all schema tables are present
- [x] All Pydantic models serialize and deserialize correctly
- [x] Repositories return typed records from the loaded IBM AMLSim dataset

---

## Phase 2 — Deterministic Tools

### Tool Registry
- [x] Implement `src/tools/registry.py` — `ToolSpec`, registration, args validation, whitelist

### Data Query Tool
- [x] Implement `src/tools/data_query/aggregator.py` — group-by, count, having, threshold logic
- [x] Implement `src/tools/data_query/tool.py` — `DataQueryTool` with audit logging

### EDA Tool
- [x] Implement `src/tools/eda/charts.py` — timeline, histogram, amount distribution (matplotlib)
- [x] Implement `src/tools/eda/tool.py` — `EDATool`: profiling, baselines, volume trends, chart paths

### Feature Engineering Tool
- [x] Implement `src/tools/features/volume.py` — `txn_count_7d/30d`, `txn_sum_7d/30d`, `avg_amount`, `amount_std`
- [x] Implement `src/tools/features/threshold.py` — `count_near_threshold_30d`, `round_number_bias`
- [x] Implement `src/tools/features/network.py` — `distinct_counterparties_30d`, `fan_in_degree`, `fan_out_degree` (NetworkX)
- [x] Implement `src/tools/features/velocity.py` — `velocity_zscore` (MAD), `dwell_time_avg_hours`, `in_out_ratio_30d`, `distinct_countries_30d`
- [x] Implement `src/tools/features/tool.py` — `FeatureTool` dispatcher, only computes requested family

### Detection Tool
- [x] Implement `src/tools/detection/rules.py` — rule detectors for structuring, smurfing, layering, rapid_cashout, velocity
- [x] Implement `src/tools/detection/ml_scorer.py` — PyOD `IForest` + `LOF` wrapper, percentile threshold, random seed from config
- [x] Implement `src/tools/detection/ensemble.py` — OR-gate combiner, triggered signals list, confidence computation
- [x] Implement `src/tools/detection/tool.py` — `DetectionTool` wiring rules + ML + ensemble

### Scoring Tool
- [x] Implement `src/tools/scoring/tool.py` — composite score formula, `RiskAssessment` output, escalation action mapping

### Explanation Tool
- [x] Implement `src/tools/explanation/templates.py` — per-pattern templates, numeric drift check (regex token diff)
- [x] Implement `src/tools/explanation/tool.py` — `ExplanationTool`, template fill, optional LLM polish (off by default in Phase 2)

### Reporting Tool
- [x] Implement `src/tools/reporting/tool.py` — `ReportingTool` assembles `AgentResponse` with execution summary, flagged items, charts, metrics

### Verification
- [x] Each tool runs end-to-end on real DuckDB data without error
- [x] Each tool's output matches its declared Pydantic schema
- [x] Audit events written for every tool call
- [x] All `tests/test_tools/` unit tests pass

---

## Phase 3 — LLM Provider Layer and Agent Glue

### LLM Providers
- [x] Implement `src/llm/providers/groq.py` — `ChatGroq` adapter
- [x] Implement `src/llm/providers/ollama.py` — `ChatOllama` adapter
- [x] Implement `src/llm/providers/openrouter.py` — `ChatOpenAI`-compatible OpenRouter adapter
- [x] Implement `src/llm/providers/lmstudio.py` — OpenAI-compatible local model adapter
- [x] Implement `src/llm/client.py` — `LLMClient` selecting adapter from `AppConfig.provider`

### Prompts
- [x] Implement `src/llm/prompts/intent_parser.py` — system prompt, few-shot examples, `QuerySpec` schema injection
- [x] Implement `src/llm/prompts/explanation_polish.py` — phrasing-only system prompt with numeric preservation instruction

### LangGraph Agent
- [x] Implement `src/agent/state.py` — `AgentState` typed dict with all shared fields
- [x] Implement `src/agent/guardrails.py` — whitelist check, args schema validation, step count cap, audit write before execution
- [x] Implement `src/agent/planner.py` — Tier 1 decision table (intent × pattern × has_entity → plan), Tier 2 LLM fallback with guardrails
- [x] Implement `src/agent/nodes.py` — node functions: `parse_intent`, `plan`, `execute_step`, `explain`, `report`
- [x] Implement `src/agent/graph.py` — `StateGraph`, conditional edges, replan-once on step failure

### Verification
- [x] Provider adapter loads from config without hardcoded credentials
- [x] `LLMClient` produces a valid `QuerySpec` from a plain text query
- [x] Guardrails reject a malformed or out-of-whitelist plan before any tool executes
- [x] LangGraph graph compiles without error
- [x] `tests/test_planner.py` golden tests pass — 3 canonical queries produce exact `tools_invoked` / `tools_skipped`
- [x] `tests/test_guardrails.py` passes


---

## Phase 4 — Query-Aware Workflows and Full Response

### Routing Paths
- [x] `pattern_search` path: `load → features → detect → score → explain → report`
- [x] `aggregation_query` path: `load → data_query → explain(lightweight) → report`
- [x] `entity_lookup` path: cache hit → `explain → report`; cache miss → `load(single) → features → detect → score → explain → report`
- [x] `broad_eda` path: `load → EDA → (optional features) → detect → score → explain → report`
- [x] `risk_scoring_batch` path: `load → features(all) → detect → score → explain → report`

### Output and API
- [x] Enable `ExplanationTool` LLM polish with numeric drift check active
- [x] `AgentResponse` fully populated: `query_spec`, `execution_summary`, `flagged_items`, `charts`, `metrics`
- [x] Implement `src/api/main.py` — FastAPI app, `POST /query` endpoint, health check, Swagger UI at `/docs`
- [x] Implement or update `main.py` — CLI entry point: `--query` flag, structured response output

### Verification
- [x] All five routing paths produce correct `AgentResponse` with correct `tools_invoked` / `tools_skipped`
- [x] Three canonical queries produce correct risk classification on IBM AMLSim synthetic data
- [x] FastAPI `POST /query` returns valid `AgentResponse` JSON
- [x] CLI `main.py --query "..."` runs end-to-end and prints structured output


---

## Phase 5 — Hardening and Review

### Bug Fixes (Session 2026-07-26)
- [x] **Bug 1 — Query-Aware Explanations**: `templates.py` now accepts `query_context` dict and prepends a query-grounded opening sentence ("In response to your query for transactions exceeding $1,000...")
- [x] **Bug 2 — Compact API Response**: `main.py` returns `AgentResponseSummary` by default (no bulk account/transaction dumps). Added `?detailed=true` and `GET /report/{query_id}/download` (JSON/CSV)
- [x] **Bug 3 — Numeric Drift False Positive**: `extract_numeric_tokens` now normalizes to float before comparison. `verify_numeric_preservation` uses set containment not list equality. LLM prompt strengthened with explicit counter-examples
- [x] **Bug 4 — Aggregation Logic**: Amount-only queries ("transactions exceeding $1000") now route to `filtered_lookup`, not HAVING aggregation
- [x] **Bug 5 — EDA-Only Path**: Visualization queries (show/chart/plot without detect/suspicious) now use EDA-only plan [EDA → REPORTING] skipping ML
- [x] **Bug 6 — Misleading Audit Steps**: Terminal steps (explain/report) no longer appear as phantom plan steps

### New Features (Session 2026-07-26)
- [x] **HBOS Third Detector**: `src/tools/detection/hbos_scorer.py` + integrated into `ml_scorer.py` and `ensemble.py`. Individual signals: `ML_IFOREST`, `ML_LOF`, `ML_HBOS`
- [x] **Semantic Cache**: `QueryCache` upgraded with TF-IDF cosine similarity (sklearn). Falls back gracefully to exact hash if sklearn unavailable
- [x] **Multithreaded Features**: `feature_family='all'` now runs 4 families in parallel via `ThreadPoolExecutor`
- [x] **Richer Tier 2 LLM Prompt**: Planner fallback prompt now includes full workflow decision guide

### Tests
- [x] `tests/test_contracts.py` — Pydantic model serialization round-trips
- [x] `tests/test_cache.py` — Exact hash, semantic TF-IDF, TTL expiry, stats
- [x] `tests/test_tools/test_explanation.py` — Template fill for all signals, numeric drift cases, query-context grounding
- [x] `tests/test_tools/test_detection.py` — IForest+LOF+HBOS scorer, HBOS standalone, ensemble OR-gate logic
- [x] `tests/test_agent/test_planner.py` — All 6 routing paths, aggregation bug fix, EDA-only path
- [x] `tests/test_tools/test_data_query.py`
- [x] `tests/test_tools/test_eda.py`
- [x] `tests/test_tools/test_features.py`
- [x] `tests/test_tools/test_scoring.py`
- [x] `tests/test_guardrails.py` — malformed plans, unknown tool names, step cap exceeded
- [x] `tests/integration/test_real_llm_*.py` — Real LLM integration tests (`Groq llama-3.3-70b-versatile`)
- [x] **ALL 304/304 TESTS PASSING (100%)**

### Logging and Audit
- [ ] Verify every tool call leaves a structured log entry with all required fields
- [ ] Verify `audit_log` table is populated correctly after an end-to-end run
- [ ] Verify no PII appears in log output or audit table

### Documentation
- [ ] Update `README.md` with quickstart, env setup, example queries, config reference, provider switching
- [x] Annotated `configs/thresholds.yaml` with field descriptions, units, safe ranges, and effect guidance
- [ ] Create `notebooks/demo.ipynb` — interactive walkthrough of the three canonical queries
