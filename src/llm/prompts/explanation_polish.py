"""Explanation polish prompt and tolerance-based numeric drift validator.

Uses an LLM to refine template-generated explanations into professional compliance prose.
A tolerance-based numeric check guards against genuine hallucination while allowing
benign LLM reformatting (e.g. "81.1" → "81.1/100", "$10,000" → "10,000 USD").
"""

import re
from loguru import logger
from llm.client import LLMClient
from tools.explanation.templates import extract_numeric_tokens

EXPLANATION_POLISH_SYSTEM_PROMPT = """You are an Anti-Money Laundering (AML) Compliance Explanation Polish Agent.
Your task is to refine raw template-generated explanations into clear, executive-ready compliance narrative prose
suitable for financial reviewers and SAR documentation.

CRITICAL NUMERIC PRESERVATION RULES — MUST FOLLOW EXACTLY:
1. Do NOT change any numerical value. Every number must appear in the output with identical digits.
   - "$10,000" must remain "$10,000" — not "$10k", not "ten thousand", not "10000"
   - "50.1%" must remain "50.1%" — not "50%", not "approximately 50%"
   - "z-score of 2.45" must remain "z-score of 2.45" — not "z-score of approximately 2"
   - "47 transactions" must remain "47 transactions" — not "nearly 50 transactions"
2. Do NOT change, abbreviate, or paraphrase account IDs, customer IDs, rule codes, or pattern names.
3. Do NOT add new facts, statistics, or inferences that are not present in the original text.
4. Do NOT remove any factual statement from the original text.
5. ONLY improve: sentence flow, professional tone, compliance vocabulary, and paragraph structure.

OUTPUT FORMAT:
- Write 2-3 concise sentences maximum.
- Use formal compliance language (e.g. "flagged for review", "exhibits indicators of", "warrants escalation").
- Retain the query context sentence if present (starts with "In response to...").
- Retain the risk level and composite score statement.

EXAMPLE (correct):
Input:  "In response to your structuring query, account 12345 was flagged. Risk level: HIGH (composite score: 82.0/100). Structuring detected: 5 transactions near the $10,000 reporting threshold with a round-number transaction bias of 60.0%."
Output: "In response to your structuring query, account 12345 was flagged as HIGH risk (composite score: 82.0/100). The account exhibits structuring indicators, with 5 transactions positioned near the $10,000 reporting threshold and a round-number transaction bias of 60.0%, warranting immediate escalation for SAR consideration."

EXAMPLE (incorrect — do NOT do this):
Input:  "risk score of 82.0"  →  Output: "risk score of approximately 82" ← WRONG: altered number
Input:  "5 transactions"      →  Output: "several transactions"           ← WRONG: removed number
Input:  "$10,000"             →  Output: "$10k"                            ← WRONG: abbreviated
"""


def _check_numeric_drift(
    grounded: str,
    candidate: str,
    tolerance: float = 0.20,
) -> tuple[bool, float, set[str]]:
    """Tolerance-based numeric drift check.

    Distinguishes between:
    - Benign reformatting: 81.1 -> 81.1 (still present in output, just rephrased around it) → OK
    - Hallucination: new numbers not in grounded text → always FAIL
    - Missing numbers: grounded numbers absent from candidate → FAIL if exceeds tolerance

    Args:
        grounded: Raw template explanation (source of truth)
        candidate: LLM-polished explanation
        tolerance: Fraction of grounded numbers allowed to be missing without triggering fallback.
                   Default 0.20 = up to 20% of numbers may be benignly reformatted.

    Returns:
        (is_acceptable, missing_ratio, hallucinated_numbers)
    """
    g_nums = set(extract_numeric_tokens(grounded))
    c_nums = set(extract_numeric_tokens(candidate))

    # Numbers in grounded but missing from candidate (dropped or reformatted away)
    missing = g_nums - c_nums
    # Numbers in candidate not present in grounded (hallucinated new values)
    hallucinated = c_nums - g_nums

    total_grounded = len(g_nums)
    missing_ratio = len(missing) / max(1, total_grounded)

    # Hallucinated numbers are always a hard fail — LLM invented facts
    if hallucinated:
        logger.debug(
            "Explanation polish: hallucinated numbers detected: {nums}",
            nums=hallucinated,
        )
        return False, missing_ratio, hallucinated

    # Missing beyond tolerance is a soft fail — too much content was dropped
    if missing_ratio > tolerance:
        return False, missing_ratio, set()

    return True, missing_ratio, set()


def polish_explanation(
    client: LLMClient,
    raw_explanation: str,
    tolerance: float = 0.20,
) -> str:
    """Polish a raw template explanation using LLM with tolerance-based drift check.

    Sends the raw explanation to the configured LLM provider with explicit
    numeric preservation instructions. Falls back to the raw template output if:
    - Any hallucinated (new, invented) number is detected in the polished text, OR
    - More than `tolerance` fraction of grounded numbers are missing.

    Benign reformatting (e.g. "81.1" still present but surrounded by different text) is accepted.

    Args:
        client: LLMClient configured with the active provider
        raw_explanation: Template-generated grounded explanation string
        tolerance: Fraction of grounded numbers allowed to be absent (default 0.20)

    Returns:
        Polished explanation string if within tolerance, else raw_explanation.
    """
    prompt = (
        "Polish the following AML compliance explanation for a formal SAR review report. "
        "Remember: preserve ALL numbers exactly as they appear.\n\n"
        f"Explanation:\n{raw_explanation}"
    )

    try:
        polished = client.invoke(prompt=prompt, system_prompt=EXPLANATION_POLISH_SYSTEM_PROMPT)
    except Exception as e:
        logger.warning("LLM explanation polish failed ({err}), falling back to raw explanation", err=e)
        return raw_explanation

    is_ok, missing_ratio, hallucinated = _check_numeric_drift(raw_explanation, polished, tolerance)

    if not is_ok:
        if hallucinated:
            logger.warning(
                "Explanation polish rejected: LLM hallucinated new numbers {nums}. "
                "Falling back to raw explanation.",
                nums=hallucinated,
            )
        else:
            logger.info(
                "Explanation polish rejected: {pct:.0f}% of grounded numbers missing (tolerance {tol:.0f}%). "
                "Falling back to raw explanation.",
                pct=missing_ratio * 100,
                tol=tolerance * 100,
            )
        return raw_explanation

    if missing_ratio > 0:
        logger.debug(
            "Explanation polish accepted with minor reformatting ({pct:.0f}% numbers rephrased, within {tol:.0f}% tolerance).",
            pct=missing_ratio * 100,
            tol=tolerance * 100,
        )
    else:
        logger.debug("Explanation polish accepted — all numerics preserved exactly.")

    return polished
