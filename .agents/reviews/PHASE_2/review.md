# Phase 2 Review

**Reviewed by**: GPT-5.4 mini
**Scope**: whole Phase 2 tool stack and tests
**Verdict**: Changes requested

## Summary
Phase 2 is broadly implemented and the test suite is strong. Several prior findings are fixed, including detection failure propagation, feature pushdown, reporting trace tracking, and explanation polish. Two core business issues still remain in the current code, though: ensemble confidence is still not computed from detectors actually run, and the ML scorer still relies on a synthetic background baseline instead of a real learned baseline.

## Major
### 🟠 Ensemble confidence still uses a fixed family denominator, `src/tools/detection/ensemble.py:47`
**Problem**: The current fix maps rule IDs to five families and divides by a fixed total of six families, regardless of which detectors were actually run for the query.
**Why it matters**: A structuring only query with one rule family and ML active can never reach confidence 1.0 even when every detector that ran agreed. The reported confidence is therefore still detached from the actual evidence available to the query.
**Suggested fix**: Track the set of detectors executed for the current plan and divide by that run set, not by a global family count.

### 🟠 ML scoring still relies on a synthetic baseline for small batches, `src/tools/detection/ml_scorer.py:41`
**Problem**: The small batch branch now augments the target accounts with 30 synthetic background rows generated from feature name heuristics instead of a real learned baseline.
**Why it matters**: The model may now produce a nonzero anomaly score for a single entity lookup, but that score is anchored to fabricated data rather than actual historical behavior. That is a correctness problem, not just a modeling preference, because it can materially distort risk ranking.
**Suggested fix**: Fit or reuse a baseline on real historical accounts, or make the synthetic baseline an explicit fallback with clear limitations and a separate code path.

### 🟡 Feature cache can drop valid cached rows on an empty slice, `src/tools/features/tool.py:127`
**Problem**: When some accounts are cached and the uncached subset has no recent transactions, the early `tx_df.empty` branch returns only synthetic zero rows for the missing accounts and discards the cached results.
**Why it matters**: A partial cache hit should not lose already computed features. This creates avoidable data loss in a real plan path for inactive accounts or quiet windows.
**Suggested fix**: Merge `cached_results` into the empty slice return path so partial cache hits remain intact.

## Strengths
- The tool contracts are consistently typed, and the registry validates inputs before dispatch.
- The test suite is substantial and the full uv run passes, which gives a solid base for tightening the business logic.
- The updated slice of Phase 2 now exercises detection failure propagation, explicit reporting trace tracking, and explanation polish behavior.

## Test coverage
Coverage is strong overall, with 129 passing tests across the tool stack. The suite now covers the previously missing failure propagation and explicit tracking branches, but it still does not prove the detector confidence semantics against the actual run set or the realism of the synthetic ML baseline.
