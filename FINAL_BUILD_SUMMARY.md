# Final Build Summary: AML Suspicious Activity Detection System

## 1. Project Overview & Architectural Vision

The **Anti-Money Laundering (AML) Suspicious Activity Detection System** is an autonomous, explainable AI agent designed for financial compliance and transaction risk analysis. Unlike rigid, fixed-pipeline compliance systems, this platform accepts natural language queries from investigators and **autonomously dynamically orchestrates** appropriate analytical tools, data lookups, feature transformations, and hybrid anomaly detection algorithms to return structured, auditable assessments.

### Core Architectural Principles
1. **Agent Orchestrates, Tools Compute**: The LangGraph agent serves purely as the reasoning and routing engine. All heavy numerical computations, SQL lookups, and feature engineering are executed within deterministic, unit-tested Python tools.
2. **Deterministic Risk & Zero LLM Hallucination in Scoring**: The LLM is invoked at exactly two points in the lifecycle—intent parsing and explanation polish. It never computes or influences anomaly scores, risk bands, or statistical thresholds.
3. **Hybrid OR-Gate Detection Ensemble**: To balance regulatory explainability with zero-day threat detection, the system combines a 9-family rule engine (catching known financial crimes) with an unsupervised Machine Learning outlier engine (catching novel patterns via PyOD). If either triggers, the account is flagged.
4. **Immutable Audit Trail**: Before any analytical tool executes, a typed `ExecutionPlan` is generated and recorded to an append-only DuckDB `audit_log` table alongside timing, row counts, and status telemetry.
5. **Strict Schema Enforcement**: Every data exchange across layers is strictly typed using Pydantic (`AccountRecord`, `TransactionRecord`, `AlertRecord`, `FeatureSet`, `AgentResponse`), preventing raw dictionary slop or silent type mismatches.

---

## 2. Technology Stack & Libraries Used

The codebase is built on **Python 3.11+** using **`uv`** as the exclusive package and dependency manager.

| Layer | Primary Libraries | Purpose in Project |
| :--- | :--- | :--- |
| **Orchestration & Agent** | `langchain`, `langchain-core`, `langgraph` | Stateful agent graphs (`StateGraph`), tool-calling loops, message history management, and conditional routing. |
| **LLM Provider Adapters** | `langchain-groq`, `langchain-ollama`, `langchain-openai`, `httpx` | Provider-neutral client layer supporting Groq, Ollama, OpenAI, OpenRouter, and local LM Studio endpoints. |
| **Data & Storage Layer** | `duckdb`, `pandas`, `numpy`, `scipy` | Embedded columnar SQL database (`activity.duckdb`) for multi-million row transaction aggregations and DataFrame processing. |
| **Machine Learning & Graph** | `pyod`, `scikit-learn`, `networkx` | Unsupervised anomaly detection (IForest, LOF, HBOS) and graph network topology metrics (fan-in, fan-out, cycles). |
| **Semantic & Exact Cache** | `fastembed` | Local lightweight vector embeddings for semantic similarity matching of paraphrased queries without external API overhead. |
| **Web API & UI** | `fastapi`, `uvicorn`, `streamlit` | Asynchronous REST API serving `/api/v1/query` and interactive investigator dashboard. |
| **Contracts & Config** | `pydantic`, `pydantic-settings`, `pyyaml`, `python-dotenv` | Schema validation, type safety, and centralized configuration loading from YAML and environment variables. |
| **Observability & QA** | `loguru`, `pytest`, `shap`, `matplotlib` | Structured JSON development logging, test suite execution (298+ tests), model interpretability, and EDA visualizations. |

---

## 3. Registered Tool Suite

The system exposes 8 specialized tool modules via `src/tools/registry.py`. Each tool validates input schemas, tracks execution duration, emits structured loguru traces, and returns a typed `ToolResult`.

| Tool Name | Module Path | Responsibilities & Capabilities |
| :--- | :--- | :--- |
| **`data_query`** | `src/tools/data_query/` | Executes parameterized DuckDB SQL queries, aggregations, and filtered time-window lookups directly on raw transaction tables without ML overhead. |
| **`eda`** | `src/tools/eda/` | Generates statistical distributions, descriptive profiles, outlier summaries, and diagnostic charts for dataset exploration. |
| **`entity_lookup`** | `src/tools/entity_lookup/` | Retrieves deep-dive profiles, alert histories, counterparties, and transaction summaries for specific account IDs. |
| **`feature_engineering`** | `src/tools/features/` | Computes rolling 7d/30d aggregations, velocity z-scores, round-number biases, structuring near-threshold counts, and NetworkX graph degrees (fan-in/out). Handles empty slices by retrieving background entities. |
| **`detection`** | `src/tools/detection/` | Evaluates 9 rule detector families (structuring, smurfing, layering, rapid cashout, velocity, dormant activation, duplicate transfers, volume spikes, multi-destination fan-out) alongside PyOD ML anomaly scorers (IForest, LOF, HBOS). |
| **`scoring`** | `src/tools/scoring/` | Synthesizes rule flags, ML outlier scores, and query context into composite risk scores `[0.0, 100.0]`, risk bands (`low`, `medium`, `high`), and confidence percentages using a dynamic ensemble denominator. |
| **`explanation`** | `src/tools/explanation/` | Generates human-readable narratives grounded in computed metrics. Applies optional LLM narrative polish guarded by numerical drift checks (<20% number alteration tolerance). |
| **`reporting`** | `src/tools/reporting/` | Formats final results into standardized, auditable `AgentResponse` Pydantic models for REST APIs, CLI output, and compliance records. |

---

## 4. Agentic Workflow & Autonomous Fallback

The agent graph (`src/agent/tool_agent.py` and `src/agent/graph.py`) implements an autonomous loop with built-in fault tolerance:

```
[Entry: parse_intent] ──(Rejected)──► [report] ──► END
          │
      (Approved)
          ▼
    [tool_agent] ◄───(Loop)───┐
          │                   │
    (Tool Calls)              │
          ▼                   │
       [tools] ───────────────┘
          │
     (No Tools / End Loop)
          ▼
   [extract_state]
          │
          ├──(Actionable Data Found)──► [explain] ──► [report] ──► END
          │
          └──(No Data / Error State)──► [fallback_fixed_flow] ──► [explain] ──► [report] ──► END
```

### Key Workflow Highlights
1. **Autonomous Tool Calling (`tool_agent`)**: The LLM receives the natural language query, parsed intent, and available tool schemas. It dynamically decides which tools to call (e.g., calling `entity_lookup` for an account query, or chaining `feature_engineering` → `detection` → `scoring` for pattern sweeps).
2. **Autonomous Fallback Routing (`fallback_fixed_flow`)**: If the LLM tool-calling loop produces an empty result, reaches iteration limits, or throws an unhandled error during tool execution, conditional edges automatically route the state to `_fallback_fixed_flow`. This fallback executes a deterministic, template-driven decision table sequence (Tier-1 Planner), ensuring 100% request completion without silent failures.
3. **Smart Query & Semantic Caching**: Before evaluating intents, the agent checks an exact-match query cache (30-min TTL) and a FastEmbed semantic embedding cache (`threshold = 0.78`). Paraphrased queries (e.g., *"Show suspicious activity in records"* ≈ *"Find suspicious activity"*) return instant cached responses.

---

## 5. Architectural & Logic Design Decisions (ADRs)

- **Why Hybrid Rule + ML (OR-Gate)?** Pure rules miss novel zero-day money laundering schemes; pure ML lacks regulatory explainability and suffers from false positives. An OR-gate ensures any explicit regulatory violation (e.g., structuring $9,900 x 3 times) triggers an immediate flag, while unsupervised Isolation Forests catch unusual high-dimensional behaviors (e.g., sudden velocity spikes combined with high graph fan-out).
- **Why DuckDB?** DuckDB provides an embedded, zero-ops, columnar SQL execution engine capable of scanning millions of financial transactions in milliseconds. It eliminates network database latency while maintaining full SQL syntax and an easy migration path to PostgreSQL for cloud production.
- **Why NetworkX at MVP Scale?** Graph databases like Neo4j introduce heavy infrastructure dependencies. For synthetic datasets up to ~500k nodes (IBM AMLSim scale), in-memory NetworkX degree and cycle calculations execute in sub-second times while integrating cleanly with NumPy/Pandas feature pipelines.
- **Why Dynamic Denominators in Confidence Scoring?** Earlier implementations divided triggered rule families by a hardcoded constant (e.g., 5), deflating confidence when running targeted queries (e.g., only checking `duplicate` transfers). The architecture now dynamically counts the exact number of evaluated detector families, producing mathematically rigorous confidence scores.

---

## 6. Runtime Defaults & Threshold Configuration

All runtime defaults are governed centrally via `AppConfig` in `src/core/config.py` and configurable via `.env` or `configs/` YAML files:

### LLM & Provider Defaults
- **Active Provider**: `GROQ` (Provider-neutral; supports `OLLAMA`, `OPENAI`, `OPENROUTER`, `LMSTUDIO`).
- **Primary Model ID**: `llama-3.3-70b-versatile` (Fast inference, strong tool-calling accuracy).
- **Fallback Model ID**: `llama-3.1-8b-instant`.
- **LLM Temperature**: `0.0` (Ensures deterministic planning and consistent tool selection).

### Domain & Threshold Defaults (`configs/thresholds.yaml`)
- **Reporting Threshold**: `$10,000.0` (Standard CTR/SAR regulatory filing limit).
- **Structuring Window**: `14 days`, **Minimum Transactions**: `3` (Catches near-threshold splitting).
- **Default Look-back Window**: `90 days`.
- **Risk Bands**: 
  - `low`: Score `0 - 39`
  - `medium`: Score `40 - 74`
  - `high`: Score `75 - 100`
- **Ensemble Weights**: Rules = `50.0%`, ML = `35.0%`, Context = `15.0%`.
- **ML Contamination**: `0.05` (Top 5% anomaly cutoff), **Percentile Cutoff**: `95.0`.
- **Explanation Polish Drift Tolerance**: `0.20` (Up to 20% of numerical values may be benignly reformatted by the LLM before triggering a rollback to deterministic text).

---

## 7. Usage Examples

### Command Line Interface (CLI)
Run natural language compliance sweeps directly from the terminal using `uv`:

```bash
# Execute a pattern query with summary output
uv run python main.py --query "Find accounts showing structuring patterns in the last 30 days" --format summary

# Run an entity lookup with JSON output and bypass caching
uv run python main.py --query "Assess risk for account ID ACCT_001234" --format json --no-cache

# Override LLM provider on the fly
uv run python main.py --query "Scan for rapid cashout anomalies" --provider ollama --model llama3
```

### REST API Server (FastAPI)
Launch the asynchronous web server for integration with front-end dashboards or compliance workflows:

```bash
# Start API server on localhost:8000
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Example API Request (cURL)**:
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "Detect multi-destination fan-out activity over the last 14 days",
           "provider": "groq",
           "use_cache": true
         }'
```

**Example JSON Response**:
```json
{
  "query_id": "run_a8f9c2e1",
  "status": "completed",
  "summary": "Detected 3 accounts exhibiting high risk multi-destination fan-out patterns.",
  "risk_band": "high",
  "flagged_items": [
    {
      "entity_id": "ACCT_889102",
      "risk_score": 88.5,
      "triggered_rules": ["R_MULTI_DEST_01", "R_VELOCITY_02"],
      "explanation": "Account transferred $45,000 to 12 distinct counterparties within 48 hours."
    }
  ],
  "execution_summary": {
    "steps_executed": 3,
    "tools_used": ["feature_engineering", "detection", "scoring"],
    "total_duration_ms": 1420.5,
    "fallback_triggered": false
  }
}
```

---

## 8. Verification & QA Status

The system is backed by an automated test suite located in `tests/`. All core contracts, tool executions, heuristics, guardrails, fallback routing, and ML baselines have been verified:

```bash
uv run pytest tests/ --ignore=tests/integration/test_real_llm_end_to_end.py
```

- **Pass Rate**: 100% (`298 passed` across unit and offline integration tests).
- **Test Coverage**: Includes edge case handling for empty DataFrame slices, single-account zero-vector anchoring in PyOD, dynamic ensemble denominator accuracy, and fallback execution routing when tool calling fails.
