"""Verification script for testing live local LLM (LM Studio) integration with LangGraph agent workflow."""

import sys
from pathlib import Path

# Add src directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config import AppConfig
from core.logging import setup_logging
from core.types import ToolName
from storage.duckdb import get_duckdb_client
from storage.repositories import AuditRepo
from tools.registry import ToolRegistry, ToolSpec
from tools.data_query.tool import execute_data_query, DataQueryInput
from tools.eda.tool import execute_eda, EDAInput
from tools.features.tool import execute_feature_engineering, FeatureInput
from tools.detection.tool import execute_detection, DetectionInput
from tools.scoring.tool import execute_scoring, ScoringInput
from tools.explanation.tool import execute_explanation, ExplanationInput
from tools.reporting.tool import execute_reporting, ReportingInput
from llm.client import LLMClient
from llm.prompts.intent_parser import parse_query_intent
from agent.graph import build_agent_graph


def setup_real_registry() -> ToolRegistry:
    """Build a ToolRegistry populated with Phase 2 tools."""
    registry = ToolRegistry()

    registry.register(ToolSpec(name=ToolName.DATA_QUERY, description="Data query tool", input_schema=DataQueryInput, callable=execute_data_query))
    registry.register(ToolSpec(name=ToolName.EDA, description="EDA tool", input_schema=EDAInput, callable=execute_eda))
    registry.register(ToolSpec(name=ToolName.FEATURE_ENGINEERING, description="Feature engineering tool", input_schema=FeatureInput, callable=execute_feature_engineering))
    registry.register(ToolSpec(name=ToolName.DETECTION, description="Detection tool", input_schema=DetectionInput, callable=execute_detection))
    registry.register(ToolSpec(name=ToolName.SCORING, description="Scoring tool", input_schema=ScoringInput, callable=execute_scoring))
    registry.register(ToolSpec(name=ToolName.EXPLANATION, description="Explanation tool", input_schema=ExplanationInput, callable=execute_explanation))
    registry.register(ToolSpec(name=ToolName.REPORTING, description="Reporting tool", input_schema=ReportingInput, callable=execute_reporting))

    return registry


def main():
    print("=== Initializing Verification ===")
    config = AppConfig.load()
    setup_logging(config)
    print(f"Active Provider: {config.provider}")
    print(f"Model ID: {config.model_id}")
    print(f"Base URL: {config.get_base_url()}")

    # 1. Test Direct LLM Client Invocation
    print("\n--- 1. Testing Local LLM Direct Invocation ---")
    try:
        client = LLMClient(config)
        test_response = client.invoke("Respond with the exact words 'LOCAL_LLM_ONLINE'")
        print(f"Direct Response: {test_response}")
    except Exception as e:
        print(f"FAILED direct LLM invocation: {e}")
        return

    # 2. Test Structured Intent Parsing
    print("\n--- 2. Testing Intent Parsing with Local LLM ---")
    query = "Find structuring patterns in the last 30 days"
    try:
        query_spec = parse_query_intent(client, query)
        print(f"Parsed QuerySpec: intent={query_spec.intent_type}, pattern={query_spec.pattern_type}")
    except Exception as e:
        print(f"FAILED intent parsing: {e}")
        return

    # 3. Test Full LangGraph Agent Workflow with Real Tools
    print("\n--- 3. Testing Full LangGraph Agent Workflow ---")
    db_client = get_duckdb_client(config)
    try:
        audit_repo = AuditRepo(db_client)
        registry = setup_real_registry()

        graph = build_agent_graph(
            config=config,
            registry=registry,
            db_client=db_client,
            audit_repo=audit_repo,
            llm_client=client,
        )

        initial_state = {"raw_query": query}
        final_state = graph.invoke(initial_state)
        print("\n--- Workflow Results ---")
        print(f"Query ID: {final_state.get('query_id')}")
        print(f"Tools Executed: {[r.tool_name for r in final_state.get('tool_results', [])]}")
        print(f"Explanation: {final_state.get('explanation')}")
        if final_state.get("agent_response"):
            print("AgentResponse created successfully.")
    except Exception as e:
        print(f"FAILED graph execution: {e}")
    finally:
        db_client.close()



if __name__ == "__main__":
    main()
