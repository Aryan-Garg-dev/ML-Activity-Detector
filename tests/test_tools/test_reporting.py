"""Tests for ReportingTool: AgentResponse assembly, FlaggedItem formatting, execution summary metrics, and end-to-end multi-tool pipeline execution."""

import pytest
from pathlib import Path

from core.config import AppConfig
from core.types import ToolName, IntentType, RiskLevel, EscalationAction
from schemas.contracts import ToolResult, AgentResponse
from storage.duckdb import DuckDBClient
from tools.registry import ToolRegistry, ToolSpec
from tools.data_query.tool import DataQueryInput, execute_data_query
from tools.eda.tool import EDAInput, execute_eda
from tools.features.tool import FeatureInput, execute_feature_engineering
from tools.detection.tool import DetectionInput, execute_detection
from tools.scoring.tool import ScoringInput, execute_scoring
from tools.explanation.tool import ExplanationInput, execute_explanation
from tools.reporting.tool import ReportingInput, execute_reporting


@pytest.fixture
def config() -> AppConfig:
    return AppConfig()


@pytest.fixture
def db_client(tmp_path: Path) -> DuckDBClient:
    """Creates a test DuckDB database with sample tables."""
    client = DuckDBClient(db_path=tmp_path / "test.duckdb")
    client.init_schema("sql/duckdb_schema.sql")

    accounts = [
        (101, "C_101", 5000.0, "US", "I", False, 1),
        (102, "C_102", 10000.0, "US", "I", True, 1),
    ]
    for acc in accounts:
        client.execute("INSERT INTO accounts VALUES (?, ?, ?, ?, ?, ?, ?)", list(acc))

    txns = [
        (1, 101, 102, "TRANSFER", 9500.0, 10, True, None),
        (2, 101, 102, "TRANSFER", 9800.0, 12, True, None),
        (3, 101, 102, "TRANSFER", 9000.0, 15, True, None),
    ]
    for tx in txns:
        client.execute("INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?)", list(tx))

    return client


# --- Reporting Tool Unit Tests ---

def test_execute_reporting_basic(db_client: DuckDBClient, config: AppConfig):
    scoring_results = {
        "risk_assessments": [
            {
                "entity_id": "101",
                "composite_score": 82.5,
                "confidence": 0.8,
                "risk_level": "high",
                "escalation_action": "report",
                "triggered_signals": ["R_STRUCT_01"],
            }
        ]
    }
    explanation_results = {
        "explanations": {"101": "Flagged as HIGH risk due to structuring activity."}
    }

    input = ReportingInput(
        raw_query="Find structuring in last 30 days",
        intent="pattern_search",
        query_id="q_rep_001",
        scoring_results=scoring_results,
        explanation_results=explanation_results,
    )
    result = execute_reporting(input, config, db_client, "q_rep_001", 1)

    assert result.status == "ok"
    assert result.tool_name == ToolName.REPORTING
    response = result.data["agent_response"]
    assert response["query_id"] == "q_rep_001"
    assert response["intent"] == "pattern_search"
    assert len(response["flagged_items"]) == 1

    item = response["flagged_items"][0]
    assert item["entity_id"] == "101"
    assert item["score"] == 82.5
    assert item["risk_level"] == "high"
    assert item["escalation_action"] == "report"


def test_execute_reporting_empty_payload_tracking(db_client: DuckDBClient, config: AppConfig):
    """Empty result payloads ({}) are correctly tracked as invoked tools, not skipped."""
    input = ReportingInput(
        raw_query="Test query",
        intent="broad_eda",
        query_id="q_rep_empty",
        eda_results={},  # Empty payload but tool was invoked!
        scoring_results={},
    )
    result = execute_reporting(input, config, db_client, "q_rep_empty", 1)
    assert result.status == "ok"
    response = result.data["agent_response"]
    summary = response["execution_summary"]
    assert str(ToolName.EDA) in summary["tools_invoked"]
    assert str(ToolName.SCORING) in summary["tools_invoked"]


def test_execute_reporting_explicit_tools_invoked(db_client: DuckDBClient, config: AppConfig):
    """Explicit tools_invoked overrides inference."""
    input = ReportingInput(
        raw_query="Explicit test query",
        intent="broad_eda",
        query_id="q_rep_explicit",
        tools_invoked=[str(ToolName.EDA), str(ToolName.REPORTING)],
    )
    result = execute_reporting(input, config, db_client, "q_rep_explicit", 1)
    assert result.status == "ok"
    response = result.data["agent_response"]
    summary = response["execution_summary"]
    assert summary["tools_invoked"] == [str(ToolName.EDA), str(ToolName.REPORTING)]


def test_reporting_langchain_export(db_client: DuckDBClient, config: AppConfig):
    spec = ToolSpec(
        name=ToolName.REPORTING,
        description="Assemble final AgentResponse",
        input_schema=ReportingInput,
        callable=execute_reporting,
    )
    lc_tool = spec.to_langchain_tool(config, db_client, "q_lc_rep")
    assert lc_tool.name == "reporting"
    assert lc_tool.args_schema == ReportingInput


# --- End-to-End Multi-Tool Workflow Test ---

def test_end_to_end_tool_pipeline(db_client: DuckDBClient, config: AppConfig, tmp_path: Path):
    """End-to-end verification running all 6 tools in sequential order.

    Pipeline: Data Query -> EDA -> Feature -> Detection -> Scoring -> Explanation -> Reporting
    """
    qid = "e2e_query_001"

    # 1. Data Query Tool
    dq_res = execute_data_query(
        DataQueryInput(table="transactions", limit=50),
        config, db_client, qid, step_id=1,
    )
    assert dq_res.status == "ok"

    # 2. EDA Tool
    eda_res = execute_eda(
        EDAInput(table="transactions", include_charts=True, chart_output_dir=str(tmp_path / "charts")),
        config, db_client, qid, step_id=2,
    )
    assert eda_res.status == "ok"

    # 3. Feature Tool
    feat_res = execute_feature_engineering(
        FeatureInput(feature_family="all", account_ids=[101, 102], use_cache=True),
        config, db_client, qid, step_id=3,
    )
    assert feat_res.status == "ok"

    # 4. Detection Tool
    det_res = execute_detection(
        DetectionInput(pattern_type="structuring", account_ids=[101, 102]),
        config, db_client, qid, step_id=4,
    )
    assert det_res.status == "ok"

    # 5. Scoring Tool
    score_res = execute_scoring(
        ScoringInput(detection_results=det_res.data, query_id=qid),
        config, db_client, qid, step_id=5,
    )
    assert score_res.status == "ok"

    # 6. Explanation Tool
    exp_res = execute_explanation(
        ExplanationInput(
            risk_assessments=score_res.data.get("risk_assessments", []),
            feature_sets=feat_res.data,
        ),
        config, db_client, qid, step_id=6,
    )
    assert exp_res.status == "ok"

    # 7. Reporting Tool (Final Assembly)
    rep_res = execute_reporting(
        ReportingInput(
            raw_query="Find structuring patterns in last 30 days",
            intent="pattern_search",
            query_id=qid,
            eda_results=eda_res.data,
            data_query_results=dq_res.data,
            feature_results=feat_res.data,
            detection_results=det_res.data,
            scoring_results=score_res.data,
            explanation_results=exp_res.data,
        ),
        config, db_client, qid, step_id=7,
    )
    assert rep_res.status == "ok"

    agent_resp = rep_res.data["agent_response"]
    assert agent_resp["query_id"] == qid
    assert len(agent_resp["execution_summary"]["tools_invoked"]) == 7
    assert len(agent_resp["flagged_items"]) > 0
    assert len(agent_resp["charts"]) > 0
    assert agent_resp["metrics"]["total_entities_assessed"] > 0
    assert len(agent_resp["explanation"]) > 0


# --- Integration Test on Real activity.duckdb ---

@pytest.mark.skipif(
    not Path("activity.duckdb").exists(),
    reason="Integration test requires real activity.duckdb",
)
class TestReportingIntegration:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = DuckDBClient(db_path="activity.duckdb")
        self.config = AppConfig()
        yield
        self.client.close()

    def test_real_dataset_full_reporting(self):
        input = ReportingInput(
            raw_query="Analyze IBM AMLSim dataset for fraud",
            intent="broad_eda",
            query_id="q_real_e2e",
        )
        result = execute_reporting(input, self.config, self.client, "q_real_e2e", 1)
        assert result.status == "ok"
        assert result.data["agent_response"]["query_id"] == "q_real_e2e"
