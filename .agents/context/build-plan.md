# Build Plan

## Overview

The build moves in four phases plus a hardening phase. Phases are sequenced so each one is independently runnable and demoable before the next begins. The deterministic core (Phases 1–2) works without any LLM. The agent layer (Phase 3) wraps the core. Query-aware routing and full response (Phase 4) completes the end-to-end system. Phase 5 hardens, tests, and documents.

---

## Phase 1 — Foundation and Contracts

**Goal:** Every other phase imports from this one. Nothing is ambiguous. Nothing is missing. The data layer is loaded and queryable. The config class is live.

**Scope:**
- `AppConfig` class: loads `.env` + `configs/app.yaml` + `configs/thresholds.yaml` + `configs/logging.yaml` in one place. Passes settings into modules. Never reads env vars inside business code.
- Structured logging setup using `loguru`. Dev-mode structured output to console and `logs/`. Log field contract documented.
- All Pydantic domain models: `AccountRecord`, `TransactionRecord`, `AlertRecord`, `FeatureSet`, `QuerySpec`, `ExecutionPlan`, `PlanStep`, `ToolResult`, `ToolCallLog`, `AuditEvent`, `RunMetadata`, `RiskAssessment`, `FlaggedItem`, `AgentResponse`
- Shared enums: `IntentType`, `PatternType`, `RiskLevel`, `EscalationAction`, `ToolName`, `ProviderName`
- DuckDB connection, schema migration (adds `feature_cache`, `audit_log`, `risk_assessments` tables), and repository layer: `AccountRepo`, `TransactionRepo`, `AlertRepo`, `AuditRepo`, `FeatureRepo`
- CSV → DuckDB ingestion script (`scripts/load_data.py`)

**Deliverables:**
1. `src/core/config.py` — `AppConfig` class
2. `src/core/logging.py` — logger setup and structured log helpers
3. `src/core/types.py` — shared enums
4. `src/schemas/domain.py` — core domain records
5. `src/schemas/contracts.py` — query, plan, tool result, response contracts
6. `src/schemas/audit.py` — audit and run metadata records
7. `src/schemas/risk.py` — risk assessment and flagged item records
8. `src/storage/duckdb.py` — connection lifecycle and query helper
9. `src/storage/repositories.py` — typed repository methods
10. `sql/duckdb_schema.sql` — updated with `feature_cache`, `audit_log`, `risk_assessments`
11. `scripts/load_data.py` — CSV ingestion and schema setup
12. `configs/app.yaml`, `configs/logging.yaml`, `configs/thresholds.yaml` — initial config files

**Exit criteria:** `AppConfig` loads without error. DuckDB connects and schema is valid. All Pydantic models serialize/deserialize correctly. Repositories return typed records from the loaded dataset.

---

## Phase 2 — Deterministic Tools

**Goal:** All ML cycle tools built and tested as standalone functions. No LLM, no agent, no routing. Each tool accepts typed input, returns typed output, logs its execution, and writes an audit event.

**Scope:**
- Tool registry (`src/tools/registry.py`): maps `ToolName` → `ToolSpec`, validates args schema on call
- `DataQueryTool`: DuckDB-backed filtered lookups and direct aggregations (group-by, count, having). No ML.
- `EDATool`: profiling summary (row count, null rates, type breakdown), transaction volume over time, amount distribution, per-segment breakdown, baseline stats. Chart output paths included in result.
- `FeatureTool`: dispatches to feature families. Computes only what the plan requests. Families: `volume`, `threshold`, `network`, `velocity`. All vectorized.
- `DetectionTool`: rule engine (`rules.py`) per pattern + PyOD ML scorer (`ml_scorer.py`) + OR-gate ensemble (`ensemble.py`). Returns raw anomaly scores and triggered rule flags.
- `ScoringTool`: composite score formula from `thresholds.yaml` weights. Emits `RiskAssessment` per entity. Confidence computed as `distinct_detectors_agreed / detectors_run`.
- `ExplanationTool`: fills pattern-specific templates with real computed values. Numeric drift check built in. LLM polish is optional and off by default at this phase.
- `ReportingTool`: assembles `AgentResponse` from all tool outputs. Formats execution summary including `tools_invoked` and `tools_skipped`.

**Deliverables:**
1. `src/tools/registry.py`
2. `src/tools/data_query/tool.py` + `aggregator.py`
3. `src/tools/eda/tool.py` + `charts.py`
4. `src/tools/features/tool.py` + `volume.py` + `threshold.py` + `network.py` + `velocity.py`
5. `src/tools/detection/tool.py` + `rules.py` + `ml_scorer.py` + `ensemble.py`
6. `src/tools/scoring/tool.py`
7. `src/tools/explanation/tool.py` + `templates.py`
8. `src/tools/reporting/tool.py`

**Exit criteria:** Each tool runs end-to-end on real DuckDB data. Each tool's output matches its declared output schema. Audit events are written for every call. All tools pass unit tests in `tests/test_tools/`.

---

## Phase 3 — LLM Provider Layer and Agent Glue

**Goal:** Provider adapters, LangChain wrappers, LangGraph graph, and planner all wired together. The agent can receive a query, build a plan, and execute tools.

**Scope:**
- Provider adapters: `groq.py`, `ollama.py`, `openrouter.py` — each returns a `BaseChatModel`. Selected by `AppConfig.provider`.
- `LLMClient` in `src/llm/client.py`: wraps the active adapter. Exposes `invoke(prompt)` and `structured_output(schema)`.
- Intent parser prompt in `src/llm/prompts/intent_parser.py` — system + few-shot examples. Output validated against `QuerySpec` schema. Re-prompt once on validation failure; surface clarifying question to user on second failure.
- Explanation polish prompt in `src/llm/prompts/explanation_polish.py` — numeric drift check enforced after LLM response.
- LangGraph state in `src/agent/state.py` — `AgentState` holds: raw query, `QuerySpec`, `ExecutionPlan`, list of `ToolResult`, list of `AuditEvent`, `AgentResponse`
- LangGraph nodes in `src/agent/nodes.py`: `parse_intent`, `plan`, `execute_step`, `explain`, `report`
- LangGraph graph in `src/agent/graph.py`: `StateGraph` connecting nodes with conditional edges; replan at most once on step failure
- Planner in `src/agent/planner.py`: Tier 1 decision table (intent × pattern × has_entity → plan template), Tier 2 LLM fallback for unmatched intents
- Guardrails in `src/agent/guardrails.py`: whitelist check, args schema validation, step cap (max 8 steps), plan written to audit before execution begins

**Deliverables:**
1. `src/llm/providers/groq.py`, `ollama.py`, `openrouter.py`
2. `src/llm/client.py`
3. `src/llm/prompts/intent_parser.py`
4. `src/llm/prompts/explanation_polish.py`
5. `src/agent/state.py`
6. `src/agent/nodes.py`
7. `src/agent/graph.py`
8. `src/agent/planner.py`
9. `src/agent/guardrails.py`

**Exit criteria:** Provider adapter loads from config. `LLMClient` produces a valid `QuerySpec` from a plain text query. Guardrails reject a malformed plan before any tool executes. LangGraph graph compiles without error. Three canonical queries each produce the correct `ExecutionPlan` (golden test asserts exact `tools_invoked` and `tools_skipped`).

---

## Phase 4 — Query-Aware Workflows and Full Response

**Goal:** End-to-end from query string to `AgentResponse`. All four routing paths work correctly: pattern search, aggregation/threshold query, entity lookup, broad EDA. Structured response includes execution trace, flagged items, explanations, escalation actions.

**Scope:**
- Five routing paths wired through the LangGraph graph:
  - `pattern_search`: load → features → detect → score → explain → report
  - `aggregation_query`: load → data_query → lightweight explain → report
  - `entity_lookup`: entity_lookup → [cache hit] explain → report; [miss] load(single) → features → detect → score → explain → report
  - `broad_eda`: load → EDA → (optional features) → detect → score → explain → report
  - `risk_scoring_batch`: load → features(all) → detect → score → explain → report
- `ExplanationTool` LLM polish enabled (with numeric drift check)
- `ReportingTool` produces `AgentResponse` with: `query_spec`, `execution_summary`, `flagged_items` (each with `composite_score`, `confidence`, `triggered_signals`, `explanation`, `recommended_action`, `supporting_evidence`), `charts`, `metrics`
- FastAPI endpoint in `src/api/main.py`: `POST /query` → `AgentResponse`
- Agent entry point in `main.py` for CLI invocation

**Deliverables:**
1. All five routing paths working in the LangGraph graph
2. `src/api/main.py` — FastAPI app
3. Updated `main.py` — CLI entry point
4. `AgentResponse` fully populated for each path

**Exit criteria:** All three canonical golden-test queries return the correct tool selection, correct risk classification on synthetic data with injected patterns, and a fully populated `AgentResponse`. FastAPI endpoint responds correctly to a POST request.

---

## Phase 5 — Hardening and Review

**Goal:** Tests cover contracts, tools, planner, and guardrails. Logging and audit events are verified. Sample queries and expected outputs are documented. The project is ready for review, demo, or handoff.

**Scope:**
- Unit tests for every tool contract (input schema, output schema, error states)
- Golden tests for the planner: 3 canonical queries assert exact `tools_invoked`/`tools_skipped`
- Guardrail tests: malformed plans and unknown tool names must be rejected
- Numeric drift tests: explanation generator rejects LLM outputs that alter numbers
- Logging validation: every tool call leaves a structured log entry with all required fields
- End-to-end test: a handful of representative queries run through the full stack and checked against `AgentResponse` contract shape
- `README.md`: quickstart, env setup, example queries, config reference, how to switch providers
- `configs/thresholds.yaml`: annotated with field descriptions and safe-to-tune guidance
- `notebooks/demo.ipynb`: interactive walkthrough of the three canonical queries with output

**Deliverables:**
1. `tests/test_contracts.py`
2. `tests/test_planner.py` (golden tests)
3. `tests/test_tools/` (one file per tool)
4. `tests/test_guardrails.py`
5. `tests/test_end_to_end.py`
6. Updated `README.md`
7. Annotated `configs/thresholds.yaml`
8. `notebooks/demo.ipynb`

**Exit criteria:** All tests pass. Logging check confirms audit fields are present. End-to-end demo runs from a clean state with a single `uv run python main.py --query "..."`.
