from enum import StrEnum


class AgentKey:
    # analyst keys
    TECHNICAL = "technical"

    COMPANY_NEWS = "company_news"
    SECTION_NEWS = "section_news"
    COMPANY_NEWS_ENHANCED = "company_news_enhanced"
    DUMMY = "dummy"
    ML_MODEL_ONLINE = "ml_model_online"
    # workflow keys
    PORTFOLIO_ENRICHED_MEMORY = "portfolio_manager_enriched_memory"
    PORTFOLIO = "portfolio_manager"
    PORTFOLIO_RISK_MANAGED = "portfolio_manager_risk_managed"


class Signal(StrEnum):
    """Signal type"""

    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"


class Action(StrEnum):
    """Action type"""

    BUY = "Buy"
    SELL = "Sell"
    HOLD = "Hold"


class NewsSection(StrEnum):
    """Fixed news section categories for classification."""

    COMPANY_FUNDAMENTALS = "company_fundamentals"
    PRODUCT_DEMAND = "product_demand"
    REGULATORY_POLICY = "regulatory_policy"
    MACRO_RATES = "macro_rates"
    INDUSTRY_COMPETITION = "industry_competition"
    FILINGS_10K_10Q = "filings_10k_10q"
    OTHER = "other"
