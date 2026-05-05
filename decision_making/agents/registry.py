from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import ClassVar

from agents.portfolio_manager import portfolio_agent, portfolio_agent_enriched_memory, portfolio_agent_risk_managed
from graph.constants import AgentKey


@dataclass(frozen=True)
class AgentSpec:
    module_path: str
    attr_name: str
    doc: str


class AgentRegistry:
    """Registry for all agents."""

    # Initialize as actual dictionaries, not just type annotations
    agent_func_mapping: ClassVar[dict[str, Callable]] = {}
    agent_doc_mapping: ClassVar[dict[str, str]] = {}
    agent_spec_mapping: ClassVar[dict[str, AgentSpec]] = {}

    # Analyst KEYs
    ANALYST_KEYS: ClassVar[list[str]] = [
        AgentKey.TECHNICAL,
        AgentKey.COMPANY_NEWS,
        AgentKey.SECTION_NEWS,
        AgentKey.DUMMY,
        AgentKey.COMPANY_NEWS_ENHANCED,
        AgentKey.ML_MODEL_ONLINE,
    ]

    @classmethod
    def get_agent_func_by_key(cls, key: str) -> Callable:
        """Get agent function by key."""
        agent_func = cls.agent_func_mapping.get(key)
        if agent_func is not None:
            return agent_func

        spec = cls.agent_spec_mapping.get(key)
        if spec is None:
            return None

        module = import_module(spec.module_path)
        agent_func = getattr(module, spec.attr_name)
        cls.agent_func_mapping[key] = agent_func
        return agent_func

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
    def register_agent(
        cls,
        key: str,
        agent_doc: str,
        agent_func: Callable | None = None,
        *,
        module_path: str | None = None,
        attr_name: str | None = None,
    ) -> None:
        """
        Register a new agent.

        Args:
            key: Unique identifier for the agent
            agent_doc: short description of the agent
        """
        cls.agent_doc_mapping[key] = agent_doc
        if agent_func is not None:
            cls.agent_func_mapping[key] = agent_func
        elif module_path is not None and attr_name is not None:
            cls.agent_spec_mapping[key] = AgentSpec(module_path=module_path, attr_name=attr_name, doc=agent_doc)
        else:
            raise ValueError("register_agent requires either agent_func or module_path/attr_name")

    @classmethod
    def run_registry(cls):
        """Run the registry."""

        cls.register_agent(
            key=AgentKey.PORTFOLIO_ENRICHED_MEMORY,
            agent_doc="Portfolio manager making final trading decisions based on the signals from the analysts and past hit rate.",
            agent_func=portfolio_agent_enriched_memory,
        )

        cls.register_agent(
            key=AgentKey.PORTFOLIO,
            agent_doc="Portfolio manager making final trading decisions based on the signals from the analysts.",
            agent_func=portfolio_agent,
        )

        cls.register_agent(
            key=AgentKey.PORTFOLIO_RISK_MANAGED,
            agent_doc="Risk-managed portfolio manager: risk control LLM call + NAV-based share sizing.",
            agent_func=portfolio_agent_risk_managed,
        )

        cls.register_agent(
            key=AgentKey.COMPANY_NEWS,
            agent_doc="Company news specialist analyzing company news and media coverage.",
            module_path="agents.analysts.company_news",
            attr_name="company_news_agent",
        )

        cls.register_agent(
            key=AgentKey.TECHNICAL,
            agent_doc="Technical analysis specialist using multiple technical analysis strategies.",
            module_path="agents.analysts.technical",
            attr_name="technical_agent",
        )

        cls.register_agent(
            key=AgentKey.SECTION_NEWS,
            agent_doc="Section-aware news analyst that classifies news into categories and scores each independently.",
            module_path="agents.analysts.section_news",
            attr_name="section_news_agent",
        )

        cls.register_agent(
            key=AgentKey.DUMMY,
            agent_doc="Dummy analyst for debugging - returns neutral signal with no analysis.",
            module_path="agents.analysts.dummy",
            attr_name="dummy_agent",
        )

        cls.register_agent(
            key=AgentKey.ML_MODEL_ONLINE,
            agent_doc="ML analyst with cross-sectional online learning: updates model daily from full SP500 cross-section before predicting the competition ticker.",
            module_path="agents.analysts.ml_model",
            attr_name="ml_model_online",
        )

        cls.register_agent(
            key=AgentKey.COMPANY_NEWS_ENHANCED,
            agent_doc="Enhanced company news analyst with per-article analysis and trend tracking.",
            module_path="agents.analysts.company_news_enhanced",
            attr_name="company_news_enhanced_agent",
        )
