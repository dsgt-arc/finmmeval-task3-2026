from enum import Enum


class AgentKey:
    # analyst keys
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    INSIDER = "insider"
    COMPANY_NEWS = "company_news"
    MACROECONOMIC = "macroeconomic"
    POLICY = "policy"
    DUMMY = "dummy"
    ML_MODEL = "ml_model"
    # workflow keys
    PORTFOLIO = "portfolio_manager"
    PLANNER = "analyst_planner"


class Signal(str, Enum):
    """Signal type"""

    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"

    def __str__(self) -> str:
        return self.value


class Action(str, Enum):
    """Action type"""

    BUY = "Buy"
    SELL = "Sell"
    HOLD = "Hold"

    def __str__(self) -> str:
        return self.value
