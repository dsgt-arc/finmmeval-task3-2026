import datetime

from graph.constants import Action, AgentKey
from graph.schema import Decision, FundState, PositionRisk
from llm.inference import agent_call
from llm.prompt import PORTFOLIO_PROMPT, RISK_CONTROL_PROMPT
from memory.manager import MemoryManager
from util.db_helper import get_db
from util.logger import logger

from decision_making.ama_data import load_specific_data

# Portfolio Manager Thresholds
thresholds = {"decision_memory_limit": 5}


def _build_analyst_memory_text(ticker: str, trading_date: str, analyst_signals: list) -> str:
    """Summarise analyst signals into a single text record for working memory."""
    parts = [f"Analyst signals for {ticker} on {trading_date}:"]
    for sig in analyst_signals:
        strength = f" (strength={sig.signal_strength:.2f})" if sig.signal_strength is not None else ""
        parts.append(f"  - {sig.signal.value}{strength}: {sig.justification[:120]}")
    return "\n".join(parts)


def portfolio_agent(state: FundState):
    """Makes final trading decisions and generates orders, leveraging layered memory."""
    agent_name = AgentKey.PORTFOLIO
    portfolio = state["portfolio"]
    ticker = state["ticker"]
    exp_name = state["exp_name"]
    trading_date = state["trading_date"]
    analyst_signals = state["analyst_signals"]
    llm_config = state["llm_config"]
    num_tickers = state["num_tickers"]

    # Get database and memory manager
    db = get_db()
    memory_manager = MemoryManager(db)

    # Get price data
    try:
        historic_date = trading_date - datetime.timedelta(days=1)
        current_price = load_specific_data(symbol=ticker, date=historic_date, type="current_price")
    except Exception as e:
        logger.error(f"Failed to fetch price data for {ticker}: {e}")
        raise RuntimeError("Failed to make decision") from e

    # ------------------------------------------------------------------
    # Step 1: Ingest today's analyst signals into working memory
    # ------------------------------------------------------------------
    date_str = trading_date.strftime("%Y-%m-%d")
    signal_text = _build_analyst_memory_text(ticker, date_str, analyst_signals)
    memory_manager.add_to_working_memory(exp_name, ticker, date_str, signal_text)
    logger.log_agent_status(agent_name, ticker, "Ingested analyst signals into working memory")

    # ------------------------------------------------------------------
    # Step 2: Query layered memory for relevant historical context
    # ------------------------------------------------------------------
    query_text = f"Trading decision for {ticker}: " + " | ".join(
        f"{s.signal.value} {s.justification[:80]}" for s in analyst_signals
    )
    working_mems, long_term_mems = memory_manager.query_memories(exp_name, ticker, query_text, top_k=6)
    memory_context = memory_manager.format_memory_context(working_mems, long_term_mems)
    logger.log_agent_status(
        agent_name,
        ticker,
        f"Retrieved {len(working_mems)} working + {len(long_term_mems)} long-term memories",
    )

    # ------------------------------------------------------------------
    # Step 3: Risk control
    # ------------------------------------------------------------------
    max_position_ratio = 1
    if num_tickers > 1:
        max_position_ratio = round(2 / num_tickers * 20) / 20

    risk_prompt = RISK_CONTROL_PROMPT.format(
        ticker_signals=analyst_signals,
        portfolio=portfolio.model_dump_json(),
        max_position_ratio=max_position_ratio,
    )

    position_risk = agent_call(
        prompt=risk_prompt,
        llm_config=llm_config,
        pydantic_model=PositionRisk,
    )

    logger.log_agent_status(agent_name, ticker, "Risk control")
    logger.log_risk(ticker, position_risk)

    if position_risk.optimal_position_ratio > max_position_ratio:
        position_risk.optimal_position_ratio = max_position_ratio
    elif position_risk.optimal_position_ratio < 0:
        position_risk.optimal_position_ratio = 0

    logger.log_agent_status(agent_name, ticker, "Making trading decisions")

    # ------------------------------------------------------------------
    # Step 4: Portfolio decision with memory context
    # ------------------------------------------------------------------
    decision_memory = db.get_decision_memory(exp_name, ticker, thresholds["decision_memory_limit"])
    current_shares, tradable_shares = calculate_ticker_shares(
        portfolio, current_price, ticker, position_risk.optimal_position_ratio
    )

    prompt = PORTFOLIO_PROMPT.format(
        decision_memory=decision_memory,
        memory_context=memory_context,
        current_price=current_price,
        current_shares=current_shares,
        tradable_shares=tradable_shares,
    )

    ticker_decision = agent_call(prompt=prompt, llm_config=llm_config, pydantic_model=Decision)

    # Post-process
    ticker_decision.price = current_price
    if ticker_decision.shares < 0 and ticker_decision.action == Action.SELL:
        ticker_decision.shares = -ticker_decision.shares

    # ------------------------------------------------------------------
    # Step 5: Save decision to DB and to working memory
    # ------------------------------------------------------------------
    logger.log_decision(ticker, ticker_decision)
    db.save_decision(portfolio.id, ticker, prompt, ticker_decision, trading_date)

    decision_text = (
        f"Decision on {date_str} for {ticker}: {ticker_decision.action.value} "
        f"{ticker_decision.shares} shares @ ${ticker_decision.price:.2f}. "
        f"Reason: {ticker_decision.justification[:150]}"
    )
    memory_manager.add_to_working_memory(exp_name, ticker, date_str, decision_text)

    # ------------------------------------------------------------------
    # Step 6: Decay & maintain memory (once per trading cycle)
    # ------------------------------------------------------------------
    memory_manager.decay_and_maintain(exp_name)
    logger.log_agent_status(agent_name, ticker, "Memory maintenance complete")

    return {"decision": ticker_decision}


def calculate_ticker_shares(portfolio, current_price, ticker, optimal_position_ratio):
    """calculate the tradable shares for a given ticker based on portfolio"""

    current_shares = 0
    if ticker in portfolio.positions:
        current_shares = portfolio.positions[ticker].shares
    current_value = current_shares * current_price
    total_portfolio_value = portfolio.cashflow + sum(portfolio.positions[t].value for t in portfolio.positions)
    position_limit = total_portfolio_value * optimal_position_ratio
    position_value_gap = position_limit - current_value

    if position_value_gap > 0:
        tradable_shares = min(position_value_gap, portfolio.cashflow) // current_price
    else:
        tradable_shares = max(position_value_gap // current_price, -current_shares)

    logger.info(
        f"Ticker {ticker}: current shares={current_shares}, current price={current_price}, "
        f"position limit=${position_limit:.2f}, position value gap=${position_value_gap:.2f}, "
        f"tradable shares={tradable_shares}"
    )
    return current_shares, tradable_shares
