import datetime

from graph.constants import AgentKey
from graph.schema import Decision, FundState
from llm.inference import agent_call
from llm.prompt import MARKET_TIMING_PROMPT
from util.db_helper import get_db
from util.logger import logger

from decision_making.ama_data import load_specific_data

thresholds = {"decision_memory_limit": 5}


def portfolio_agent(state: FundState):
    """Market timing agent — outputs Buy/Sell/Hold based on analyst signals."""
    agent_name = AgentKey.PORTFOLIO
    ticker = state["ticker"]
    exp_name = state["exp_name"]
    trading_date = state["trading_date"]
    analyst_signals = state["analyst_signals"]
    llm_config = state["llm_config"]

    db = get_db()

    try:
        historic_date = trading_date - datetime.timedelta(days=1)
        current_price = load_specific_data(symbol=ticker, date=historic_date, type="current_price")
    except Exception as e:
        logger.error(f"Failed to fetch price data for {ticker}: {e}")
        raise RuntimeError("Failed to make decision")

    decision_memory = db.get_decision_memory(exp_name, ticker, thresholds["decision_memory_limit"])

    n_bullish = sum(1 for s in analyst_signals if str(s.signal).lower() == "bullish")
    n_bearish = sum(1 for s in analyst_signals if str(s.signal).lower() == "bearish")
    signal_balance = n_bullish - n_bearish

    prompt = MARKET_TIMING_PROMPT.format(
        ticker=ticker,
        analyst_signals=analyst_signals,
        signal_balance=f"{signal_balance:+d} ({n_bullish} bullish, {n_bearish} bearish)",
        decision_memory=decision_memory,
        current_price=current_price,
    )

    ticker_decision = agent_call(prompt=prompt, llm_config=llm_config, pydantic_model=Decision)

    ticker_decision.price = current_price
    # Normalise shares to symbolic values: 1 for directional, 0 for hold
    if ticker_decision.action.value in ("Buy", "Sell"):
        ticker_decision.shares = 1
    else:
        ticker_decision.shares = 0

    logger.log_agent_status(agent_name, ticker, "Market timing decision made")
    logger.log_decision(ticker, ticker_decision)

    return {"decision": ticker_decision}
