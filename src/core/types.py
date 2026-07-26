from enum import StrEnum

# Enum defining supported user query intent types
class IntentType(StrEnum):
    PATTERN_SEARCH = "pattern_search"
    AGGREGATION_QUERY = "aggregation_query"
    ENTITY_LOOKUP = "entity_lookup"
    BROAD_EDA = "broad_eda"
    RISK_SCORING_BATCH = "risk_scoring_batch"

# Enum defining AML suspicious activity patterns
class PatternType(StrEnum):
    STRUCTURING = "structuring"
    SMURFING = "smurfing"
    LAYERING = "layering"
    RAPID_CASHOUT = "rapid_cashout"
    VELOCITY = "velocity"
    UNKNOWN = "unknown"

# Enum defining risk levels for assessed entities
class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# Enum defining recommended compliance actions
class EscalationAction(StrEnum):
    MONITOR = "monitor"
    REVIEW = "review"
    REPORT = "report"

# Enum defining tools available in the agent registry
class ToolName(StrEnum):
    DATA_QUERY = "data_query"
    EDA = "eda"
    FEATURE_ENGINEERING = "feature_engineering"
    DETECTION = "detection"
    SCORING = "scoring"
    EXPLANATION = "explanation"
    REPORTING = "reporting"
    ENTITY_LOOKUP = "entity_lookup"

# Enum defining supported LLM backend providers
class ProviderName(StrEnum):
    GROQ = "groq"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    LMSTUDIO = "lmstudio"
    OPENAI = "openai"

