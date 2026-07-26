## Project identity
This repository builds an agentic AML (Anti-Money Laundering) suspicious activity detection system. The system accepts a natural language query, autonomously decides what analysis to perform, selects and invokes the minimum required tools from the ML cycle, and returns a structured, explainable, auditable result. The agent is not a fixed pipeline. It routes between tools based on parsed query intent.

**Domain:** Financial compliance, anti-money laundering, transaction risk detection.
**Dataset:** IBM AMLSim synthetic data. Three tables in `activity.duckdb`: `accounts`, `transactions`, `alerts`. Source CSVs in `dataset/`.
**Stack:** Python 3.11+ · DuckDB · pandas · PyOD (IForest, LOF) · NetworkX · LangChain · LangGraph · loguru · Pydantic · FastAPI.
**Context files:** `.agents/context/` — architecture, build plan, progress tracker, project overview, and this file expanded.

## Project goals
1. Keep the code modular and easy to extend.
2. Keep the runtime minimal and avoid slop.
3. Keep logging clear in development.
4. Keep the model provider swappable.
5. Keep the data layer explicit and auditable.
6. Keep tool contracts typed and stable.

## Core design rules
1. The agent orchestrates, tools compute.
2. Every major action is a tool.
3. No hidden business logic inside prompts.
4. No hardcoded provider choice inside business modules.
5. No raw sensitive data to the model when a derived field will do.
6. No silent fallback paths that skip audit logging.
7. Prefer deterministic code for scoring, thresholds, and routing whenever possible.
8. The ExecutionPlan is written to audit_log before the first tool executes.
9. Every tool returns a typed Pydantic model. No raw dicts or strings between components.
10. The LLM is used at exactly two points: intent parsing (QuerySpec) and explanation polish. It never computes a score.

## Implementation Rules
1. Strictly follow the implementation plan for every module. Use the prescribed responsibilities, architecture, folder structure, interfaces, and recommended libraries. Do not redesign the architecture unless there is a compelling engineering reason, and if you do, explain the reasoning before making any changes.
2. Write production-quality code only. The code should be modular, maintainable, scalable, reusable, and easy to understand. Use proper Python type hints throughout the project. Use dataclasses or Pydantic models where appropriate. Follow SOLID principles wherever applicable. Avoid shortcuts, temporary implementations, unnecessary duplication, or poor design decisions that would make future modules difficult to build.
3. Write clean, concise and meaningful comments throughout the codebase. Comments should explain business logic, algorithms, assumptions, architectural decisions, complex implementations, and non-obvious code paths. Avoid commenting every line or writing redundant comments. The goal is to help another engineer understand and maintain the project. Comments should not exceed 1-2 lines.
4. Use centralized logging throughout the application instead of print statements. Add logging wherever it is useful, including configuration loading, application startup, dataset loading, validation, caching, feature engineering, model training, inference, recommendation generation, orchestration, warnings, handled exceptions, and unexpected states. Use appropriate logging levels including DEBUG, INFO, WARNING, ERROR, and CRITICAL.
5. Implement robust error handling. Validate all inputs before processing. Raise meaningful custom exceptions instead of generic ones whenever appropriate. Never silently ignore errors. Include informative error messages that help with debugging.
6. Always use uv for dependency management. Never use pip or pip install commands. Whenever a dependency is required, use commands such as "uv add pandas", "uv add scikit-learn", or "uv add pydantic". Assume the project is managed entirely using uv, including virtual environments and package installation.
7. Every module must include a comprehensive test suite before it is considered complete. Mirror the project structure inside the tests directory. Use pytest for all testing. Write unit tests, integration tests where appropriate, edge case tests, invalid input tests, exception tests, regression tests, and compatibility tests with previously completed modules. Do not leave any public functionality untested. Aim for high test coverage and production-level confidence in the implementation.
8. After completing the implementation of the selected modules, perform a verification pass. Check that imports are correct, there are no circular dependencies, type hints are consistent, interfaces match previous modules, logging has been added where appropriate, documentation is present, tests are comprehensive, and the implementation follows the architecture defined in the implementation plan. Clearly mention any known limitations or future improvements if they exist.
9. Never modify previously completed modules unless a genuine bug or architectural issue is discovered. If an interface must change, explain why the change is necessary and preserve backward compatibility whenever possible.
10. Follow good repository standards. Whenever appropriate, update supporting files such as pyproject.toml, README.md, .env.example, configuration files, linting configuration, formatting configuration, and testing configuration. Maintain a consistent project structure throughout the repository.
11. Prioritize correctness over optimization. After correctness, prioritize maintainability, readability, modularity, testability, scalability, and finally performance. Do not perform premature optimization
12. At the end of every iteration, provide a concise progress report listing the modules completed, files created or modified, tests added, dependencies introduced using uv and any architectural decisions made.

## Runtime shape
Use `AppConfig` in `src/core/config.py` as the single source of truth for all runtime settings. It loads from `.env` and all files in `configs/`. Pass `AppConfig` into modules rather than reading env vars anywhere else. All threshold values, rule parameters, provider choice, model IDs, data paths, and logging settings live in config.

## Data and model contract
Core domain records: `AccountRecord`, `TransactionRecord`, `AlertRecord`, `FeatureSet`, `QuerySpec`, `ExecutionPlan`, `PlanStep`, `ToolResult`, `AuditEvent`, `RunMetadata`, `RiskAssessment`, `FlaggedItem`, `AgentResponse`. All defined as Pydantic models in `src/schemas/`. Pass typed models between all layers.

## Tool map
| Area | Tool Module | What it covers | Notes |
|---|---|---|---|
| Data query | `src/tools/data_query/` | DuckDB and SQL style lookups, aggregations | Use for direct answers and filtered slices; no ML |
| EDA | `src/tools/eda/` | Profiles, distributions, summaries, charts | Run only when the query needs exploration |
| Feature engineering | `src/tools/features/` | Rolling, frequency, threshold, network, and deviation features | Compute only the requested feature family |
| Detection | `src/tools/detection/` | Rule engine + PyOD ML scorer + OR-gate ensemble | Hybrid: rules catch known patterns, ML catches the rest |
| Scoring | `src/tools/scoring/` | Composite score → risk band + confidence | Deterministic and config-driven |
| Explanation | `src/tools/explanation/` | Human-readable reasons for flags | Grounded in computed values; LLM polish is optional and drift-checked |
| Reporting | `src/tools/reporting/` | Final structured AgentResponse assembly | Shapes result for reviewer and audit |

## Provider strategy
The LLM layer must stay provider neutral. Support Groq, Ollama, OpenRouter, or another LangChain-compatible backend through `src/llm/client.py` and adapters in `src/llm/providers/`. Provider names, base URLs, model IDs, and retry settings belong in `AppConfig`. Business code only imports `LLMClient`, never a provider-specific class.

## Logging and audit rules
Use loguru for structured development logs. Every tool call logs: `query_id`, `tool_name`, `step_id`, `started_at`, `ended_at`, `duration_ms`, `rows_in`, `rows_out`, `provider`, `model_name`, `config_version`, `status` (`ok | error | skipped`), `error_summary`. Errors also log the stack trace. Audit events are written to the `audit_log` DuckDB table in addition to log output. Audit log is append-only. No PII in logs.

## Coding practices
Keep modules small. Keep names specific. Keep public interfaces narrow. Prefer Pydantic models for all contracts. Prefer pure functions for transforms. Avoid deep nesting. Avoid magic constants — use `AppConfig` and `configs/thresholds.yaml`. Add tests for contracts and planner behavior before expanding scope to the next phase.

## Build phases
| Phase | Focus | Key Deliverable |
|---|---|---|
| 1 | Foundation and contracts | `AppConfig`, all Pydantic models, DuckDB layer, loguru setup |
| 2 | Deterministic tools | All tools built, unit-tested, audit-logged — no LLM needed |
| 3 | LLM provider layer and agent glue | Provider adapters, LangGraph graph, planner, guardrails |
| 4 | Query-aware workflows and full response | All 5 routing paths, FastAPI endpoint, CLI entry point |
| 5 | Hardening and review | Test suite, logging validation, README, demo notebook |

Full details in `.agents/context/build-plan.md`. Checklist in `.agents/context/progress-tracker.md`.

## Key architecture decisions
- **ADR-1:** Template-driven Plan-and-Execute (decision table + validated LLM fallback), not free-form ReAct.
- **ADR-2:** LangGraph for the agent graph (StateGraph with typed state, conditional edges, replan-once on failure).
- **ADR-3:** Hybrid rule + ML ensemble (OR-gate). Rules explain; ML catches novel patterns. AND-gate would miss novel threats.
- **ADR-4:** DuckDB as the primary store. Zero-ops, SQL-native, columnar, clean migration path to Postgres.
- **ADR-5:** LLM used at exactly two points: intent parsing and explanation polish. Never computes a score.
- **ADR-6:** NetworkX for graph features (fan-in/out, cycles) at IBM AMLSim scale. Revisit if graph exceeds ~500k nodes.
- **ADR-7:** No DBSCAN (third detector adds hyperparameter surface without labels to tune). No XGBoost (requires labeled data). No Neo4j at MVP scale.

## Relevant skills
| Skill | When to use it | What it should govern |
|---|---|---|
| `architect` | When a design choice is still open | Project structure, stack, and build plan |
| `develop` | When implementation starts | Feature delivery against the agreed plan |
| `debug` | When a tool or flow breaks | Root cause analysis and focused fixes |
| `test` | When code or tools are added | Regression and contract tests |
| `check` | Before merge or handoff | Verification and review quality |
| `sync` | After the change is stable | Keep repo notes and scope aligned |
| `scope` | When planning the next slice | Feature boundaries and sequencing |
| `scikit-learn` | For model work | Model selection, training, and evaluation patterns |
| `exploratory-data-analysis` | For CSV and tabular review | Dataset profiling and quality checks |
| `matplotlib` | For charts | Simple review plots and diagnostics |
| `shap` | For explanations | Model interpretation and feature contribution views |
| `od-expert` | For anomaly detection work | Unsupervised detection workflow and comparison |

