"""Tests for ScoringTool: composite score calculation, risk level mapping, DuckDB persistence, and LangChain tool export."""

import pytest
from pathlib import Path

from core.config import AppConfig
from core.types import ToolName, RiskLevel, EscalationAction
from storage.duckdb import DuckDBClient
from tools.registry import ToolRegistry, ToolSpec
from tools.scoring.tool import ScoringInput, execute_scoring, _classify_risk


@pytest.fixture
def config() -> AppConfig:
    return AppConfig()


@pytest.fixture
def db_client(tmp_path: Path) -> DuckDBClient:
    client = DuckDBClient(db_path=tmp_path / "test.duckdb")
    client.init_schema("sql/duckdb_schema.sql")
    return client


# --- Unit Tests: Risk Classification ---

def test_classify_risk_bands(config: AppConfig):
    # Low risk (0 - 39)
    risk, esc = _classify_risk(25.0, config)
    assert risk == RiskLevel.LOW
    assert esc == EscalationAction.MONITOR

    # Medium risk (40 - 74)
    risk, esc = _classify_risk(55.0, config)
    assert risk == RiskLevel.MEDIUM
    assert esc == EscalationAction.REVIEW

    # High risk (75 - 100)
    risk, esc = _classify_risk(85.0, config)
    assert risk == RiskLevel.HIGH
    assert esc == EscalationAction.REPORT


# --- Unit Tests: Execute Scoring ---

def test_execute_scoring_basic(db_client: DuckDBClient, config: AppConfig):
    detection_data = {
        "detection_results": {
            "101": {
                "triggered_signals": ["R_STRUCT_01"],
                "rule_flags": ["R_STRUCT_01"],
                "ml_anomaly_score": 0.3,
                "confidence": 0.5,
            },
            "102": {
                "triggered_signals": ["R_STRUCT_01", "R_STRUCT_02", "ML_ANOMALY"],
                "rule_flags": ["R_STRUCT_01", "R_STRUCT_02"],
                "ml_anomaly_score": 0.95,
                "confidence": 1.0,
            },
        }
    }
    input = ScoringInput(detection_results=detection_data, query_id="q_score_001")
    result = execute_scoring(input, config, db_client, "q_score_001", 1)

    assert result.status == "ok"
    assert result.tool_name == ToolName.SCORING
    assert result.data["assessed_count"] == 2

    assessments = result.data["risk_assessments"]
    assert len(assessments) == 2

    acc102_assess = next(a for a in assessments if a["entity_id"] == "102")
    assert acc102_assess["risk_level"] in [RiskLevel.MEDIUM, RiskLevel.HIGH]
    assert acc102_assess["composite_score"] > 50.0


def test_execute_scoring_empty_input(db_client: DuckDBClient, config: AppConfig):
    input = ScoringInput(detection_results={}, query_id="q_score_empty")
    result = execute_scoring(input, config, db_client, "q_score_empty", 1)
    assert result.status == "ok"
    assert result.data["assessed_count"] == 0


def test_execute_scoring_persists_to_duckdb(db_client: DuckDBClient, config: AppConfig):
    detection_data = {
        "detection_results": {
            "201": {
                "triggered_signals": ["R_VELOCITY_01"],
                "rule_flags": ["R_VELOCITY_01"],
                "ml_anomaly_score": 0.8,
                "confidence": 0.8,
            }
        }
    }
    input = ScoringInput(detection_results=detection_data, query_id="q_score_db")
    execute_scoring(input, config, db_client, "q_score_db", 1)

    rows = db_client.query("SELECT entity_id, composite_score, risk_level FROM risk_assessments WHERE entity_id = '201'")
    assert len(rows) == 1
    assert rows[0][0] == "201"
    assert rows[0][1] > 0


# --- LangChain Export Test ---

def test_scoring_langchain_export(db_client: DuckDBClient, config: AppConfig):
    spec = ToolSpec(
        name=ToolName.SCORING,
        description="Calculate composite AML risk score",
        input_schema=ScoringInput,
        callable=execute_scoring,
    )
    lc_tool = spec.to_langchain_tool(config, db_client, "q_lc_score")
    assert lc_tool.name == "scoring"
    assert lc_tool.args_schema == ScoringInput


# --- Integration Test on Real activity.duckdb ---

@pytest.mark.skipif(
    not Path("activity.duckdb").exists(),
    reason="Integration test requires real activity.duckdb",
)
class TestScoringIntegration:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = DuckDBClient(db_path="activity.duckdb")
        self.config = AppConfig()
        yield
        self.client.close()

    def test_real_dataset_scoring(self):
        detection_data = {
            "detection_results": {
                "0": {"triggered_signals": ["R_STRUCT_01"], "rule_flags": ["R_STRUCT_01"], "ml_anomaly_score": 0.5, "confidence": 0.5},
            }
        }
        input = ScoringInput(detection_results=detection_data, query_id="q_int_score_001")
        result = execute_scoring(input, self.config, self.client, "q_int_score_001", 1)
        assert result.status == "ok"
        assert result.data["assessed_count"] == 1
