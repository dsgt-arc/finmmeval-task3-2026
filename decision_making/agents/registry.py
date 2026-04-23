from collections.abc import Callable

from agents.analysts import *
from agents.portfolio_manager import portfolio_agent
from graph.constants import AgentKey


class AgentRegistry:
    """Registry for all agents."""

    # Initialize as actual dictionaries, not just type annotations
    agent_func_mapping: dict[str, Callable] = {}
    agent_doc_mapping: dict[str, str] = {}

    # Analyst KEYs
    ANALYST_KEYS = [
        AgentKey.TECHNICAL,
        AgentKey.FUNDAMENTAL,
        AgentKey.INSIDER,
        AgentKey.COMPANY_NEWS,
        AgentKey.MACROECONOMIC,
        AgentKey.POLICY,
        AgentKey.DUMMY,
        AgentKey.COMPANY_NEWS_ENHANCED,
        AgentKey.ML_MODEL_ONLINE,
    ]

    @classmethod
    def get_agent_func_by_key(cls, key: str) -> Callable:
        """Get agent function by key."""
        return cls.agent_func_mapping.get(key)

    @classmethod
    def get_all_analyst_keys(cls) -> list[str]:
        """Get all analyst keys."""
        return cls.ANALYST_KEYS

    @classmethod
    def check_agent_key(cls, key: str) -> bool:
        """Check if an agent key is valid."""
        return key in cls.ANALYST_KEYS

    @classmethod
    def get_analyst_info(cls, key: str) -> str:
        """Get analyst info."""
        return cls.agent_doc_mapping[key]

    @classmethod
    def register_agent(cls, key: str, agent_func: Callable, agent_doc: str) -> None:
        """
        Register a new agent.

        Args:
            key: Unique identifier for the agent
            agent_func: Function that implements the agent logic
            agent_doc: short description of the agent
        """
        cls.agent_func_mapping[key] = agent_func
        cls.agent_doc_mapping[key] = agent_doc

    @classmethod
    def run_registry(cls):
        """Run the registry."""

        cls.register_agent(
            key=AgentKey.PORTFOLIO,
            agent_func=portfolio_agent,
            agent_doc="Portfolio manager making final trading decisions based on the signals from the analysts.",
        )

        cls.register_agent(
            key=AgentKey.COMPANY_NEWS,
            agent_func=company_news_agent,
            agent_doc="Company news specialist analyzing company news and media coverage.",
        )

        cls.register_agent(
            key=AgentKey.TECHNICAL,
            agent_func=technical_agent,
            agent_doc="Technical analysis specialist using multiple technical analysis strategies.",
        )

        cls.register_agent(
            key=AgentKey.DUMMY,
            agent_func=dummy_agent,
            agent_doc="Dummy analyst for debugging - returns neutral signal with no analysis.",
        )

        cls.register_agent(
            key=AgentKey.ML_MODEL_ONLINE,
            agent_func=ml_model_agent_online,
            agent_doc="ML analyst with cross-sectional online learning: updates model daily from full SP500 cross-section before predicting the competition ticker.",
        )

        cls.register_agent(
            key=AgentKey.COMPANY_NEWS_ENHANCED,
            agent_func=company_news_enhanced_agent,
            agent_doc="Enhanced company news analyst with per-article analysis and trend tracking.",
        )
