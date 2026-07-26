# Phase 1 Review

## Verdict
Phase 1 is **partially complete**. The foundation code exists and the current uv test suite passes, but I would not mark the phase as fully complete against the stated spec yet. The main gaps are config precedence, log contract fidelity, and weak verification of the actual exit criteria.

## Validation
- `uv run pytest -q` passes: 7 tests passed.
- The existing tests only cover a small happy-path slice of the foundation layer.

## Findings

### 1. `AppConfig.load()` does not actually apply `.env` values, so runtime config precedence is broken.
`src/core/config.py` loads `.env` via `load_dotenv`, but the constructor only receives merged YAML content. There is no step that reads from `os.environ`, so any setting placed in `.env` is ignored unless it is also duplicated in YAML. That contradicts the phase 1 runtime contract that config should load `.env` plus YAML in one place.

Evidence: [src/core/config.py](../../src/core/config.py#L47-L62)

### 2. Structured tool logging is emitted as a formatted string, not as structured log fields.
`log_tool_execution()` builds a payload dictionary, but then writes it with `logger.info(f"TOOL_EXECUTION: {json.dumps(payload)}")`. That makes downstream parsing dependent on string scraping instead of a structured sink, which weakens the stated audit/log contract and makes validation of required fields harder.

Evidence: [src/core/logging.py](../../src/core/logging.py#L27-L58)

### 3. The current tests do not prove phase 1 exit criteria; they mostly re-check defaults and happy paths.
`tests/test_config.py` only asserts values that are already duplicated in `AppConfig` defaults, so it would still pass if the YAML files were never read. `tests/test_storage.py` only exercises `AccountRepo`, `AuditRepo`, and `FeatureRepo` on a handcrafted record; it does not cover `TransactionRepo`, `AlertRepo`, `scripts/load_data.py`, or any schema/bootstrap edge cases. As a result, the test suite gives a green signal without proving that phase 1’s data-loading and repository guarantees hold on the real dataset.

Evidence: [tests/test_config.py](../../tests/test_config.py#L6-L12), [tests/test_storage.py](../../tests/test_storage.py#L8-L44)

## Additional Notes
- The phase 1 deliverables are present on disk, and the current uv-managed tests are green.
- `scripts/load_data.py` is present, but there is no test coverage for ingestion repeatability or for the sentinel handling of `ALERT_ID = -1` on the real CSV inputs.
- The repository also contains a second ingestion script, `scripts/load_duckdb.py`, which is closer to a repeatable load flow than `scripts/load_data.py`; that suggests some duplication/legacy drift in the phase 1 surface.

## Bottom Line
I would treat phase 1 as **implementation complete but not fully verified**. Before calling it done, I would fix env-driven config loading, make the logging helper emit genuinely structured events, and add tests that prove the config/load/repository contracts against the actual phase 1 acceptance criteria.

## Follow-Up Verification

The requested fixes were applied and the current uv-managed tests pass.

- `src/core/config.py` now reads environment variables and lets them override YAML-backed settings.
- `src/core/logging.py` now writes structured tool-execution records to `tool_execution.jsonl` using bound Loguru context.
- `configs/logging.yaml` now exposes the new console/file logging toggles and per-sink levels.
- `tests/test_config.py` now covers custom YAML loading, env overrides, structured JSONL output, and the console fallback path.
- `tests/test_storage.py` now covers `TransactionRepo`, `AlertRepo`, and a dataset ingestion path in addition to the existing repositories.

Validation run: `uv run pytest -q tests/test_config.py tests/test_storage.py tests/test_models.py` -> 10 passed.

Residual review note: the new console-fallback test currently exercises setup without asserting the actual fallback handler behavior, so it improves coverage but does not fully prove that branch yet.
