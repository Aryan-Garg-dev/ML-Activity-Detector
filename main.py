"""CLI entry point for Anti-Money Laundering (AML) Suspicious Activity Detector Agent."""

import sys
import argparse
import json
from pathlib import Path

# Ensure src/ is in sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.config import AppConfig
from core.types import ProviderName
from core.cache import QueryCache
from storage.duckdb import get_duckdb_client
from storage.repositories import AuditRepo
from tools.registry import build_default_tool_registry
from llm.client import LLMClient
from agent.graph import build_agent_graph


def parse_args():
    parser = argparse.ArgumentParser(description="AML Suspicious Activity Detector AI Agent CLI")
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="Find structuring patterns in the last 30 days",
        help="Natural language compliance query string.",
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        choices=[p.value for p in ProviderName],
        default=None,
        help="Override active LLM provider choice.",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Override model ID string.",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["json", "summary"],
        default="summary",
        help="Output format: json (raw Pydantic JSON) or summary (formatted text).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Bypass the smart query cache and force fresh computation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = AppConfig.load()
    if args.provider:
        config.provider = ProviderName(args.provider)
    if args.model:
        config.model_id = args.model

    db_client = get_duckdb_client(config)

    # Initialize smart cache
    cache = QueryCache(
        ttl_minutes=config.query_cache_ttl_minutes,
        enabled=config.query_cache_enabled and not args.no_cache,
    )

    try:
        # Check cache first
        cached_response = cache.get(args.query)
        if cached_response is not None:
            agent_response = cached_response
            print("[Cache HIT] Returning cached result.")
        else:
            audit_repo = AuditRepo(db_client)
            registry = build_default_tool_registry()
            llm_client = LLMClient(config)

            graph = build_agent_graph(
                config=config,
                registry=registry,
                db_client=db_client,
                audit_repo=audit_repo,
                llm_client=llm_client,
            )

            initial_state = {"raw_query": args.query}
            final_state = graph.invoke(initial_state)

            agent_response = final_state.get("agent_response")
            if not agent_response:
                print("ERROR: Agent workflow did not produce an AgentResponse.", file=sys.stderr)
                sys.exit(1)

            # Store in cache for future runs
            cache.put(args.query, agent_response)

        if args.format == "json":
            print(json.dumps(agent_response.model_dump(), indent=2))
        else:
            print("\n========================================================")
            print(" AML SUSPICIOUS ACTIVITY DETECTOR - AGENT RESPONSE")
            print("========================================================")
            print(f"Query ID:        {agent_response.query_id}")
            print(f"Raw Query:       {agent_response.raw_query}")
            print(f"Intent Type:     {agent_response.intent}")
            print("\n--- Execution Summary ---")
            summary = agent_response.execution_summary
            print(f"Tools Invoked:   {', '.join(summary.get('tools_invoked', []))}")
            print(f"Tools Skipped:   {', '.join(summary.get('tools_skipped', []))}")
            print(f"Steps Executed:  {summary.get('steps_executed', 0)}")
            print("\n--- Summary Metrics ---")
            for k, v in agent_response.metrics.items():
                print(f"  {k}: {v}")

            print("\n--- Grounded Compliance Narrative ---")
            print(agent_response.explanation)

            print("\n--- Flagged Suspicious Items ---")
            if not agent_response.flagged_items:
                print("  No suspicious accounts or activities flagged.")
            else:
                for idx, item in enumerate(agent_response.flagged_items[:5], 1):
                    eid = item.get("entity_id")
                    score = item.get("score")
                    level = item.get("risk_level")
                    act = item.get("escalation_action")
                    acc = item.get("account_record")
                    txs = item.get("recent_transactions", [])
                    print(f"\n  [{idx}] Entity ID: {eid} | Score: {score:.1f}/100 | Risk: {level.upper()} | Action: {act.upper()}")
                    if acc:
                        print(f"      Account Details: Customer={acc.get('customer_id')}, Type={acc.get('account_type')}, Country={acc.get('country')}, InitBalance=${acc.get('init_balance', 0):,.2f}")
                    if txs:
                        print(f"      Recent Transactions: {len(txs)} txs loaded (latest: ${txs[0].get('tx_amount', 0):,.2f} at {txs[0].get('timestamp')})")
                    print(f"      Explanation: {item.get('explanation')}")

            print("\n========================================================\n")

    finally:
        db_client.close()


if __name__ == "__main__":
    main()
