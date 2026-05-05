from time import perf_counter
from typing import Any

from agents.registry import AgentRegistry
from graph.constants import AgentKey
from graph.schema import Action, Decision, FundState, Portfolio, Position
from langgraph.graph import END, START, StateGraph
from util.db_helper import get_db
from util.logger import logger


class AgentWorkflow:
    """Trading Decision Workflow."""

    def __init__(self, config: dict[str, Any], config_id: str):
        self.llm_config = config["llm"]
        self.tickers = config["tickers"]
        self.exp_name = config["exp_name"]
        self.trading_date = config["trading_date"]
        self.db = get_db()

        self.portfolio_mode = config.get("portfolio_mode", "market_timing")

        if self.portfolio_mode == "risk_managed":
            portfolio = self.db.get_latest_portfolio(config_id)
            if not portfolio:
                portfolio = self.db.create_portfolio(config_id, config["cashflow"], config["trading_date"])
                if not portfolio:
                    raise RuntimeError(f"Failed to create portfolio for config {self.exp_name}")
            new_portfolio = self.db.copy_portfolio(config_id, portfolio, config["trading_date"])
            self.init_portfolio = Portfolio(**new_portfolio)
            logger.info(f"New portfolio ID: {self.init_portfolio.id}")
        else:
            portfolio_id = self.db.create_portfolio_stub(config_id, config["trading_date"])
            if not portfolio_id:
                raise RuntimeError(f"Failed to create portfolio stub for config {self.exp_name}")
            self.portfolio_id = portfolio_id
            logger.info(f"Portfolio stub ID: {self.portfolio_id}")

        self.sequential_mode = config.get("sequential_mode", False)
        self.enriched_memory = config.get("enriched_memory", False)
        self.api_payload = config.get("api_payload")

        if not config.get("workflow_analysts"):
            raise ValueError("workflow_analysts must be provided in config")

        self.workflow_analysts = config["workflow_analysts"]
        invalid_analysts = [a for a in self.workflow_analysts if not AgentRegistry.check_agent_key(a)]
        if invalid_analysts:
            logger.warning(f"Invalid analyst keys removed: {invalid_analysts}")
            self.workflow_analysts = [a for a in self.workflow_analysts if a not in invalid_analysts]

        if not self.workflow_analysts:
            raise ValueError("No valid analysts remaining after validation")

    def build(self) -> StateGraph:
        """Build the workflow."""
        graph = StateGraph(FundState)

        if self.portfolio_mode == "risk_managed":
            portfolio_key = AgentKey.PORTFOLIO_RISK_MANAGED
        elif self.enriched_memory:
            portfolio_key = AgentKey.PORTFOLIO_ENRICHED_MEMORY
        else:
            portfolio_key = AgentKey.PORTFOLIO
        portfolio_func = AgentRegistry.get_agent_func_by_key(portfolio_key)
        graph.add_node(portfolio_key, portfolio_func)

        prev_node = START
        if self.sequential_mode:
            for analyst in self.current_analysts:
                agent_func = AgentRegistry.get_agent_func_by_key(analyst)
                graph.add_node(analyst, agent_func)
                graph.add_edge(prev_node, analyst)
                prev_node = analyst
            graph.add_edge(prev_node, portfolio_key)
        else:
            for analyst in self.current_analysts:
                agent_func = AgentRegistry.get_agent_func_by_key(analyst)
                graph.add_node(analyst, agent_func)
                graph.add_edge(prev_node, analyst)
                graph.add_edge(analyst, portfolio_key)

        graph.add_edge(portfolio_key, END)
        workflow = graph.compile()
        logger.info(
            "Workflow graph compiled exp_name=%s sequential=%s analysts=%s",
            self.exp_name,
            self.sequential_mode,
            self.current_analysts,
        )

        return workflow

    def load_analysts(self, ticker: str):
        """Load analysts for the current ticker."""
        logger.info("Using all verified analysts")
        self.current_analysts = self.workflow_analysts.copy()

        logger.info(f"Active analysts for {ticker}: {self.current_analysts}")

    def run(self, config_id: str) -> float:
        """Run the workflow."""
        start_time = perf_counter()
        logger.info(
            "Workflow run start config_id=%s exp_name=%s tickers=%s sequential_mode=%s",
            config_id,
            self.exp_name,
            self.tickers,
            self.sequential_mode,
        )

        portfolio = self.init_portfolio if self.portfolio_mode == "risk_managed" else None

        for ticker in self.tickers:
            logger.info("Workflow ticker start ticker=%s config_id=%s", ticker, config_id)
            self.load_analysts(ticker)

            if self.portfolio_mode == "risk_managed":
                state = FundState(
                    ticker=ticker,
                    exp_name=self.exp_name,
                    trading_date=self.trading_date,
                    llm_config=self.llm_config,
                    portfolio_id=portfolio.id,
                    portfolio=portfolio,
                    num_tickers=len(self.tickers),
                    api_payload=self.api_payload,
                )
            else:
                state = FundState(
                    ticker=ticker,
                    exp_name=self.exp_name,
                    trading_date=self.trading_date,
                    llm_config=self.llm_config,
                    portfolio_id=self.portfolio_id,
                    portfolio=None,
                    num_tickers=None,
                    api_payload=self.api_payload,
                )

            workflow = self.build()
            logger.info(f"{ticker} workflow compiled successfully")
            try:
                final_state = workflow.invoke(state)
            except Exception as e:
                logger.exception("Error running deep fund ticker=%s config_id=%s", ticker, config_id)
                raise RuntimeError(f"Failed to run workflow for {ticker}") from e

            decision = final_state["decision"]
            logger.info(
                "Workflow ticker complete ticker=%s action=%s shares=%s price=%s",
                ticker,
                decision.action,
                decision.shares,
                decision.price,
            )

            if self.portfolio_mode == "risk_managed":
                portfolio = self.update_portfolio_ticker(portfolio, ticker, decision)
                logger.log_portfolio(f"{ticker} position update", portfolio)
            else:
                self.db.save_decision(self.portfolio_id, ticker, "market_timing", decision, self.trading_date)

        if self.portfolio_mode == "risk_managed":
            logger.log_portfolio("Final Portfolio", portfolio)
            logger.info("Updating portfolio to Database")
            self.db.update_portfolio(config_id, portfolio.model_dump(), self.trading_date)

        end_time = perf_counter()
        time_cost = end_time - start_time

        return time_cost

    def update_portfolio_ticker(self, portfolio: Portfolio, ticker: str, decision: Decision) -> Portfolio:
        """Update the ticker asset in the portfolio."""

        action = decision.action
        shares = decision.shares
        price = decision.price

        if ticker not in portfolio.positions:
            portfolio.positions[ticker] = Position(shares=0, value=0)

        if action == Action.BUY:
            portfolio.positions[ticker].shares += shares
            portfolio.cashflow -= price * shares
        elif action == Action.SELL:
            total_portfolio_value = portfolio.cashflow + sum(pos.value for pos in portfolio.positions.values())
            max_short = int(max(0, total_portfolio_value) // price)
            max_sell = portfolio.positions[ticker].shares + max_short
            actual_sell = min(shares, max(0, max_sell))
            portfolio.positions[ticker].shares -= actual_sell
            portfolio.cashflow += price * actual_sell

        # Always recalculate position value with latest price
        portfolio.positions[ticker].value = round(price * portfolio.positions[ticker].shares, 2)

        # round cashflow
        portfolio.cashflow = round(portfolio.cashflow, 2)

        return portfolio
