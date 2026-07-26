"""Intent parser prompt and schema binding for AML natural language queries."""

from schemas.contracts import QuerySpec
from core.types import IntentType, PatternType
from llm.client import LLMClient

INTENT_PARSER_SYSTEM_PROMPT = """You are an Anti-Money Laundering (AML) Compliance Query Intent Parser.
Your task is to parse natural language queries from compliance analysts and extract structured intent specification (QuerySpec).

Input Query Intent Types:
- pattern_search: User wants to detect specific AML suspicious activity patterns (structuring, smurfing, layering, rapid_cashout, velocity) across transactions/accounts.
- aggregation_query: User wants simple counting, sums, or threshold lookups (e.g. "counts of transactions over $10,000", "customers with 10+ transactions under $10,000"). ML detection is NOT needed.
- entity_lookup: User asks about a specific customer or account ID (e.g. "Is customer ID 4521 suspicious?").
- broad_eda: User wants general exploration, profiling, baseline stats, or charts on the dataset.
- risk_scoring_batch: User wants batch risk scoring across all accounts or entities.

Pattern Types:
- structuring: Multiple transactions just under reporting threshold ($10,000).
- smurfing: Many counterparty accounts transferring funds into one central account.
- layering: Complex movement of funds through multiple intermediary accounts.
- rapid_cashout: Funds transferred into an account and immediately withdrawn.
- velocity: Abnormal spike in transaction frequency or total volume over baseline.
- unknown: When query does not specify a distinct pattern.

Few-Shot Examples:
Query: "Find structuring patterns in the last 30 days"
QuerySpec: {
  "intent_type": "pattern_search",
  "pattern_type": "structuring",
  "target_entity_id": null,
  "filters": {"days": 30, "max_amount": 9999.0},
  "aggregation_spec": {},
  "raw_query": "Find structuring patterns in the last 30 days"
}

Query: "Which customers made 10+ transactions under $10,000?"
QuerySpec: {
  "intent_type": "aggregation_query",
  "pattern_type": "structuring",
  "target_entity_id": null,
  "filters": {"max_amount": 9999.0},
  "aggregation_spec": {"min_count": 10, "group_by": "account_id"},
  "raw_query": "Which customers made 10+ transactions under $10,000?"
}

Query: "Is customer ID 4521 suspicious?"
QuerySpec: {
  "intent_type": "entity_lookup",
  "pattern_type": "unknown",
  "target_entity_id": "4521",
  "filters": {},
  "aggregation_spec": {},
  "raw_query": "Is customer ID 4521 suspicious?"
}

Query: "Give me an overview of overall transaction volume trends and distributions"
QuerySpec: {
  "intent_type": "broad_eda",
  "pattern_type": "unknown",
  "target_entity_id": null,
  "filters": {},
  "aggregation_spec": {},
  "raw_query": "Give me an overview of overall transaction volume trends and distributions"
}
"""


def parse_query_intent(client: LLMClient, raw_query: str) -> QuerySpec:
    """Parse a natural language query into a typed QuerySpec using LLMClient."""
    prompt = f"Parse the following AML compliance query:\n\n\"{raw_query}\""
    query_spec = client.structured_output(
        schema=QuerySpec,
        prompt=prompt,
        system_prompt=INTENT_PARSER_SYSTEM_PROMPT,
    )
    # Ensure raw_query is retained
    query_spec.raw_query = raw_query
    return query_spec
