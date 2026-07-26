# Project Overview

## Problem Statement

Financial institutions are required by regulatory bodies (FinCEN, FATF) to run Anti-Money Laundering (AML) compliance programs. Traditional rule-based systems generate excessive false positives, overwhelming compliance teams, while sophisticated laundering techniques (structuring, smurfing, layering, rapid cash-out) evade static rule engines.

The challenge is to build an **agentic** suspicious activity detection system that autonomously decides what analysis to perform, selects and invokes the right tools from the ML cycle, and returns a structured, explainable, auditable result — not a fixed pipeline.

---

## What the System Is

An autonomous agent that:

1. Accepts a natural language query from an analyst
2. Parses intent, extracts filters (date range, segment, country, transaction type), identifies target entities and AML pattern types
3. Builds a dynamic execution plan — not a fixed sequential pipeline
4. Invokes only the tools needed to answer that specific query
5. Returns a structured response with: risk scores, grounded explanations, escalation recommendations, and an execution trace

The defining constraint: **adaptive tool selection**. The same tool set invoked in different combinations based on query intent. A broad query triggers EDA and full detection. A narrow count query skips ML entirely. A single entity lookup skips full-dataset profiling.

---

## What the Agent Must Do

| Capability | Description |
|---|---|
| Parse intent | Extract intent type, AML pattern, filters, entities from natural language |
| Plan dynamically | Build an execution plan — which tools, in what order, on which data subset |
| Run data query | Answer direct aggregation or threshold questions via DuckDB |
| Run EDA | Profile distributions, volume trends, baselines — only when needed |
| Engineer features | Compute rolling sums, frequency, velocity, threshold proximity, network features on demand |
| Detect anomalies | Hybrid rule + ML (IForest + LOF) detection; rules are deterministic, ML catches the rest |
| Classify risk | Convert anomaly scores and rule flags into low / medium / high risk bands via config |
| Explain flags | Ground every explanation in actual computed values; LLM only polishes phrasing |
| Recommend action | Monitor / Flag for review / Report — tied to risk level and confidence |
| Return trace | Every query produces an execution trace: what ran, what was skipped, and why |

---

## Three Canonical Query Examples

**Query A — Pattern search:** `"Find structuring patterns in the last 30 days"`
- Plan: `load_data → feature_engineering(structuring) → anomaly_detection(hybrid) → risk_classification → explanation`
- Skipped: EDA, aggregation tool

**Query B — Aggregation:** `"Which customers made 10+ transactions under $10,000?"`
- Plan: `load_data → aggregation_tool(direct SQL)`
- Skipped: feature engineering, ML detection, EDA entirely

**Query C — Entity lookup:** `"Is customer ID 4521 suspicious?"`
- Plan: `entity_lookup → [cache hit] return; [cache miss] load_data(single entity) → feature_engineering → anomaly_detection → risk_classification → explanation`
- Skipped: EDA, batch scan, aggregation

---

## AML Patterns the System Must Cover

| Pattern | Core Signal |
|---|---|
| Structuring | N+ transactions in a rolling window, each below the reporting threshold, just under the line |
| Smurfing | High fan-in/fan-out degree — multiple depositors funneling into one account |
| Layering | Chains of transfers with short dwell time between hops, often cross-jurisdiction |
| Rapid cash-out | Large withdrawal shortly after deposit, inconsistent with customer history |
| Velocity | Transaction count/volume significantly exceeds the customer's own historical baseline |

---

## Dataset

The project uses the IBM AMLSim synthetic dataset (already loaded into `activity.duckdb`). Three tables:

- `accounts` — account metadata, initial balance, country, account type, fraud label
- `transactions` — sender/receiver accounts, amount, timestamp, type, fraud label, alert_id
- `alerts` — denormalized alert records joining transaction and account context

---

## Core Constraints

- **LLM plans and explains; deterministic code computes.** Scores, thresholds, and classification are never LLM inference.
- **No fixed pipeline.** The agent routes between tools based on intent; not every query runs every tool.
- **Provider neutral.** Groq, Ollama, OpenRouter, or any LangChain-compatible backend must be switchable via config.
- **Minimal, modular, no slop.** Modules are small. Interfaces are narrow. No hidden logic in prompts.
- **Auditable.** Every tool call logs query_id, tool name, step, duration, row counts, model/provider used, and result status.
- **PII minimization.** Only pseudonymous IDs and derived feature values reach the LLM. Never raw names or account numbers.
- **Config-driven.** Risk thresholds, rule parameters, provider choice, paths, and logging settings all live in a central config class — never hardcoded or scattered across modules.
