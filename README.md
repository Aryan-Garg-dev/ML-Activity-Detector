# AML Suspicious Activity Detector

An autonomous AI-powered agent for Anti-Money Laundering (AML) suspicious activity detection. Accepts natural language compliance queries, dynamically plans and executes tool pipelines, and returns structured, explainable, auditable risk assessments.

→ **Full architecture and sample workflow:** [docs/project_overview.md](./docs/project_overview.md)

---

## Quickstart

### 1. Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- A configured LLM provider (Groq recommended — free tier available)

### 2. Clone & Install

```bash
git clone <repo-url>
cd ML-Activity-Detector
uv sync
```

### 3. Environment Setup

Copy the example env file and fill in your provider credentials:

```bash
cp .env.example .env
```

Minimum required for Groq (default provider):

```env
GROQ_API_KEY=gsk_your_key_here
PROVIDER=groq
MODEL_ID=llama-3.3-70b-versatile
```

### 4. Load Dataset

```bash
uv run python scripts/load_data.py
```

This loads the IBM AMLSim CSVs from `dataset/` into `activity.duckdb`.

### 5. Run the API Server

```bash
uv run uvicorn api.main:app --app-dir src --reload --host 127.0.0.1 --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

### 6. Run the CLI

```bash
uv run python main.py --query "Find structuring patterns in the last 30 days"
uv run python main.py --query "Is customer 4521 suspicious?" --format json
uv run python main.py --query "Which accounts made 10+ transactions under $10,000?"
```

---

## Example Queries

| Query | Intent | Tools Used |
|---|---|---|
| `"Find structuring patterns in the last 30 days"` | pattern_search | features, detection, scoring |
| `"Is customer 4521 suspicious?"` | entity_lookup | entity_lookup, scoring |
| `"Which accounts made 10+ transactions under $10,000?"` | aggregation_query | data_query |
| `"Show me the distribution of transaction amounts"` | broad_eda | data_query, eda |
| `"Score all accounts for risk"` | risk_scoring_batch | features, detection, scoring |
| `"Detect money laundering with amounts exceeding $50,000"` | pattern_search | features, detection, scoring |
| `"Is there any suspicious activity?"` | pattern_search | features, detection, scoring |

---

## API Reference

### `POST /query`

Run a natural language compliance query through the agent pipeline.

**Request:**
```json
{
  "query": "Find structuring patterns in the last 30 days",
  "provider": "groq",
  "model_id": "llama-3.3-70b-versatile"
}
```

**Response (compact, default):**
```json
{
  "query_id": "q_a1b2c3d4",
  "intent": "pattern_search",
  "explanation": "In response to your structuring pattern detection query, account 9998 was flagged...",
  "flagged_items": [
    {
      "entity_id": "9998",
      "composite_score": 81.1,
      "risk_level": "high",
      "escalation_action": "file_sar",
      "triggered_signals": ["R_STRUCT_01", "R_STRUCT_02"],
      "explanation": "..."
    }
  ],
  "metrics": {"accounts_analyzed": 1500, "flagged_count": 3}
}
```

**Options:**
- `?detailed=true` — include full `account_record` and `recent_transactions` in each flagged item
- `GET /report/{query_id}/download` — download full JSON or CSV report

### `GET /health`

Service health check with DuckDB connectivity status.

### `GET /audit/{query_id}`

Retrieve audit events for a specific query execution.

---

## Switching LLM Providers

The agent is provider-neutral. Override in `.env` or per-request in the API:

```env
# Groq (cloud, fast, free tier)
PROVIDER=groq
GROQ_API_KEY=gsk_...
MODEL_ID=llama-3.3-70b-versatile

# Ollama (local, no API key needed)
PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
MODEL_ID=llama3.1:8b

# LM Studio (local, OpenAI-compatible)
PROVIDER=lmstudio
LMSTUDIO_BASE_URL=http://localhost:1234/v1
MODEL_ID=mistralai/ministral-3-3b

# OpenRouter (cloud, 100+ models)
PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
MODEL_ID=meta-llama/llama-3.1-70b-instruct
```

See [docs/LLM_LOCAL_INTEGRATION_GUIDE.md](./docs/LLM_LOCAL_INTEGRATION_GUIDE.md) for detailed local model setup.

---

## Configuration

All runtime settings are in `src/core/config.py` (loaded from `.env` + `configs/`).
Key settings in `configs/app.yaml`:

| Setting | Default | Description |
|---|---|---|
| `semantic_cache_threshold` | `0.78` | Cosine similarity threshold for semantic cache hits |
| `explain_top_n_default` | `5` | Number of flagged accounts shown in pattern search results |
| `explain_top_n_entity` | `1` | Accounts shown for entity lookup queries |
| `explain_top_n_batch` | `10` | Accounts shown for batch risk scoring |
| `explanation_polish_tolerance` | `0.20` | Fraction of numbers allowed to be benignly reformatted by LLM polish |
| `default_window_days` | `90` | Default look-back window for open-ended date queries |
| `reporting_threshold` | `10000.0` | BSA reporting threshold used in structuring detection |

---

## Running Tests

```bash
uv run pytest tests/ -v
```

Expected: **304+ tests passing**.

Run integration tests only:
```bash
uv run pytest tests/integration/ -v
```

---

## Project Structure

```
src/
├── agent/          # LangGraph graph, nodes, planner, guardrails, state
├── api/            # FastAPI application (main.py)
├── core/           # AppConfig, cache (QueryCache), types
├── llm/            # LLM client, provider adapters, intent parser, explanation polish
├── schemas/        # All Pydantic domain models (contracts, audit, risk)
├── storage/        # DuckDB client, repositories
└── tools/          # Tool implementations
    ├── data_query/
    ├── detection/
    ├── eda/
    ├── entity_lookup/
    ├── explanation/
    ├── features/
    ├── reporting/
    └── scoring/
configs/            # app.yaml, thresholds.yaml, logging.yaml
dataset/            # IBM AMLSim CSV files
docs/               # project_overview.md, LLM_LOCAL_INTEGRATION_GUIDE.md
tests/              # Unit + integration test suites
main.py             # CLI entry point
```
