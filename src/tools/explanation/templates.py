"""Pattern-specific grounded explanation templates and numeric preservation guardrails.

Generates human-readable, auditable explanations grounded in computed features and signals.
Includes a numeric preservation check using regex token extraction to detect and reject
numeric drift in downstream LLM text polish.
"""

import re
from typing import Any
from core.types import PatternType


def _build_query_context_sentence(
    entity_id: str,
    query_context: dict[str, Any] | None,
) -> str:
    """Build an opening sentence that grounds the explanation in the user's original query.

    Examples:
      "In response to your query for transactions exceeding $1,000, account 12345 was flagged."
      "Responding to your structuring pattern search, account 12345 was flagged."
    """
    if not query_context:
        return ""

    raw_query = str(query_context.get("raw_query", ""))
    filters = query_context.get("filters", {}) or {}
    pattern = str(query_context.get("pattern_type", "")).replace("_", " ")

    # Build filter phrase from detected query filters
    filter_parts: list[str] = []
    if "min_amount" in filters and filters["min_amount"] is not None:
        filter_parts.append(f"exceeding ${float(filters['min_amount']):,.0f}")
    if "max_amount" in filters and filters["max_amount"] is not None:
        filter_parts.append(f"under ${float(filters['max_amount']):,.0f}")
    if "days" in filters and filters["days"] is not None:
        filter_parts.append(f"within the last {filters['days']} days")

    filter_phrase = ", ".join(filter_parts)

    if pattern and pattern not in ("unknown", ""):
        context_phrase = f"your {pattern} pattern detection query"
    elif filter_phrase:
        context_phrase = f"your query for transactions {filter_phrase}"
    elif raw_query:
        # Truncation limit from caller (query_context key) — fallback: 80 chars
        max_chars: int = int(query_context.get("query_display_max_chars", 80))
        q_short = raw_query[:max_chars] + "..." if len(raw_query) > max_chars else raw_query
        context_phrase = f'your query "{q_short}"'
    else:
        return ""

    return f"In response to {context_phrase}, account {entity_id} was flagged."


def format_grounded_explanation(
    entity_id: str,
    risk_info: dict[str, Any],
    feature_info: dict[str, Any] | None = None,
    pattern: PatternType | str = PatternType.UNKNOWN,
    query_context: dict[str, Any] | None = None,
) -> str:
    """Generate grounded, evidence-backed explanation text for a flagged entity.

    Args:
        entity_id: Account or customer ID string
        risk_info: Dict containing composite_score, risk_level, triggered_signals, etc.
        feature_info: Dict of computed feature values for the entity
        pattern: Target AML pattern type
        query_context: Dict with raw_query, filters, intent_type, pattern_type from QuerySpec.
                       When provided, the explanation opens with a sentence grounded in the
                       user's original query intent and applied filters.

    Returns:
        Structured human-readable explanation string grounded in both query context and evidence.
    """
    feature_info = feature_info or {}
    signals = risk_info.get("triggered_signals", [])
    score = float(risk_info.get("composite_score", 0.0))
    risk_level = str(risk_info.get("risk_level", "medium")).upper()

    # Query-grounded opening sentence — ties explanation to user's request
    context_sentence = _build_query_context_sentence(entity_id, query_context)

    explanations: list[str] = []

    # Pattern-specific grounded explanations based on triggered signals and features
    if "R_STRUCT_01" in signals or "R_STRUCT_02" in signals:
        near_cnt = int(feature_info.get("count_near_threshold_30d", 0))
        round_bias = float(feature_info.get("round_number_bias", 0.0))
        explanations.append(
            f"Structuring detected: {near_cnt} transactions near the $10,000 reporting threshold "
            f"with a round-number transaction bias of {round_bias * 100:.1f}%."
        )

    if "R_SMURF_01" in signals or "R_SMURF_02" in signals:
        fan_in = int(feature_info.get("fan_in_degree", 0))
        fan_out = int(feature_info.get("fan_out_degree", 0))
        explanations.append(
            f"Smurfing behavior: fan-in degree of {fan_in} senders "
            f"and fan-out degree of {fan_out} receivers."
        )

    if "R_LAYER_01" in signals or "R_LAYER_02" in signals:
        counterparties = int(feature_info.get("distinct_counterparties_30d", 0))
        countries = int(feature_info.get("distinct_countries_30d", 1))
        io_ratio = float(feature_info.get("in_out_ratio_30d", 0.0))
        explanations.append(
            f"Layering pattern: funds moved across {counterparties} counterparties in {countries} countries "
            f"with an in-out volume ratio of {io_ratio:.2f}."
        )

    if "R_CASHOUT_01" in signals:
        dwell_hours = float(feature_info.get("dwell_time_avg_hours", 0.0))
        io_ratio = float(feature_info.get("in_out_ratio_30d", 0.0))
        explanations.append(
            f"Rapid cash-out: average fund holding time of {dwell_hours:.1f} hours "
            f"with an in-out ratio of {io_ratio:.2f}."
        )

    if "R_VELOCITY_01" in signals:
        zscore = float(feature_info.get("velocity_zscore", 0.0))
        explanations.append(
            f"Transaction velocity anomaly: robust z-score of {zscore:.2f} (significantly above population baseline)."
        )

    # Individual ML detector signals (IForest, LOF, HBOS) — more specific than generic ML_ANOMALY
    ml_signals = [s for s in signals if s in ("ML_IFOREST", "ML_LOF", "ML_HBOS", "ML_ANOMALY")]
    if ml_signals:
        ml_score = float(risk_info.get("ml_anomaly_score", feature_info.get("ml_anomaly_score", 0.0)))
        detector_names = "/".join(s.replace("ML_", "") for s in ml_signals) if ml_signals[0] != "ML_ANOMALY" else "IForest/LOF"
        
        feature_contributions = risk_info.get("feature_contributions", {})
        contrib_text = ""
        if feature_contributions:
            items = [f"{k} (+{v})" for k, v in feature_contributions.items()]
            contrib_text = f" Top contributing features: {', '.join(items)}."

        explanations.append(
            f"Unsupervised ML ({detector_names}) flagged this account "
            f"at anomaly rank {ml_score:.2f}.{contrib_text}"
        )

    if "R_DORMANT_01" in signals or "R_DORMANT_02" in signals:
        txn_7d = int(feature_info.get("txn_count_7d", 0))
        dwell = float(feature_info.get("dwell_time_avg_hours", 0.0))
        explanations.append(
            f"Dormant reactivation: {txn_7d} transactions in last 7 days "
            f"after average inter-transaction interval of {dwell:.1f} hours."
        )

    if "R_DUPLICATE_01" in signals:
        round_bias = float(feature_info.get("round_number_bias", 0.0))
        txn_30d = int(feature_info.get("txn_count_30d", 0))
        explanations.append(
            f"Duplicate transfer pattern: {round_bias * 100:.1f}% round-number bias "
            f"across {txn_30d} transactions in 30 days."
        )

    if "R_SPIKE_01" in signals or "R_SPIKE_02" in signals:
        sum_7d = float(feature_info.get("txn_sum_7d", 0.0))
        sum_30d = float(feature_info.get("txn_sum_30d", 0.0))
        explanations.append(
            f"Volume spike: ${sum_7d:,.2f} in 7-day volume "
            f"against ${sum_30d:,.2f} total 30-day volume."
        )

    if "R_MULTI_DEST_01" in signals or "R_MULTI_DEST_02" in signals:
        counterparties = int(feature_info.get("distinct_counterparties_30d", 0))
        fan_out = int(feature_info.get("fan_out_degree", 0))
        explanations.append(
            f"Multi-destination dispersal: funds distributed to {counterparties} distinct counterparties "
            f"with a fan-out degree of {fan_out}."
        )

    # Fallback when no specific pattern rule matched
    if not explanations:
        sig_str = ", ".join(signals) if signals else "elevated feature activity"
        explanations.append(
            f"Flagged as {risk_level} risk (score: {score:.1f}/100) due to: {sig_str}."
        )

    entity_prefix = f"Account {entity_id}: " if entity_id and str(entity_id).lower() not in ("unknown", "") else ""
    summary_header = f"{entity_prefix}Risk level: {risk_level} (composite score: {score:.1f}/100)."
    evidence_text = " ".join(explanations)

    # Assemble: context sentence first (if any), then header and evidence
    if context_sentence:
        return f"{context_sentence} {summary_header} {evidence_text}"
    return f"{summary_header} {evidence_text}"


def extract_numeric_tokens(text: str) -> list[str]:
    """Extract and normalize all numeric values from text for drift comparison.

    Strips currency symbols, commas, and percent signs before matching.
    All values are converted to float and rounded to 4 decimal places so that
    "$10,000" and "10000.0" and "10000" all produce "10000.0", eliminating
    false-positive drift rejections from benign LLM reformatting.
    """
    cleaned = re.sub(r"[$,£€%]", "", text)
    matches = re.findall(r"\b\d+(?:\.\d+)?\b", cleaned)
    normalized: list[str] = []
    for m in matches:
        try:
            val = round(float(m), 4)
            normalized.append(str(val))
        except ValueError:
            normalized.append(m)
    return sorted(normalized)


def verify_numeric_preservation(grounded_text: str, candidate_text: str) -> bool:
    """Verify that candidate polished text preserves all numeric facts from grounded text.

    Uses set-based containment: all numbers from grounded_text must be present in
    candidate_text. This is intentionally lenient — candidate_text may omit redundant
    repetitions, but must not lose or alter any specific data value.

    Equivalent reformats are accepted: "$10,000" == "10000", "50.0%" == "50%".

    Returns:
        True if all grounded numeric values appear in candidate text.
        False only if a specific grounded number is clearly missing.
    """
    g_nums = set(extract_numeric_tokens(grounded_text))
    c_nums = set(extract_numeric_tokens(candidate_text))

    # Any grounded number missing from candidate is a drift violation
    missing = g_nums - c_nums
    return len(missing) == 0
