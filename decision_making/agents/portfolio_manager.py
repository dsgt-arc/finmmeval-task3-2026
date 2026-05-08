import datetime

from graph.constants import Action, AgentKey
from graph.schema import Decision, FundState, PositionRisk
from llm.inference import agent_call
from llm.prompt import MARKET_TIMING_PROMPT, MARKET_TIMING_PROMPT_W_MEMORY, PORTFOLIO_PROMPT, RISK_CONTROL_PROMPT
from util.db_helper import get_db
from util.logger import logger

from decision_making.ama_data import load_specific_data

thresholds = {
    "decision_memory_limit": 10,
    "hit_rate_min_sample": 3,
    "hit_rate_conservative_threshold": 40,
}


def _compute_decision_hit_rate(decision_memory: list[dict], ticker: str) -> dict:
    """Enrich past decisions with profitability labels and compute rolling hit-rate."""
    enriched = []
    hits, total = 0, 0

    for d in decision_memory:
        action = d.get("action", "")
        entry_price = d.get("price")
        trading_date_raw = d.get("trading_date")

        if action not in ("Buy", "Sell") or not entry_price or not trading_date_raw:
            enriched.append({**d, "outcome": "N/A"})
            continue

        try:
            if isinstance(trading_date_raw, str):
                outcome_date = datetime.datetime.fromisoformat(trading_date_raw)
            else:
                outcome_date = datetime.datetime(trading_date_raw.year, trading_date_raw.month, trading_date_raw.day)
            outcome_price = load_specific_data(symbol=ticker, date=outcome_date, type="current_price")

            is_hit = (action == "Buy" and outcome_price > entry_price) or (action == "Sell" and outcome_price < entry_price)
            hits += int(is_hit)
            total += 1
            enriched.append({**d, "outcome": "HIT" if is_hit else "MISS", "outcome_price": outcome_price})
        except Exception:
            enriched.append({**d, "outcome": "unknown"})

    hit_rate_pct = (hits / total * 100) if total > 0 else None
    return {"enriched_memory": enriched, "hits": hits, "total": total, "hit_rate_pct": hit_rate_pct}


def _run_portfolio(state: FundState, enriched_memory: bool):
    """Shared portfolio logic."""
    ticker = state["ticker"]
    exp_name = state["exp_name"]
    trading_date = state["trading_date"]
    analyst_signals = state["analyst_signals"]
    llm_config = state["llm_config"]
    api_payload = state.get("api_payload")
    market_date = trading_date if api_payload else trading_date - datetime.timedelta(days=1)

    db = get_db()

    try:
        current_price = load_specific_data(
            symbol=ticker,
            date=market_date,
            type="current_price",
            api_payload=api_payload,
        )
    except Exception as e:
        logger.error(f"Failed to fetch price data for {ticker}: {e}")
        raise RuntimeError("Failed to make decision") from e

    decision_memory = db.get_decision_memory(exp_name, ticker, thresholds["decision_memory_limit"])
    if enriched_memory:
        hit_rate_result = _compute_decision_hit_rate(decision_memory, ticker)
        total = hit_rate_result["total"]
        hit_rate_pct = hit_rate_result["hit_rate_pct"]
        if total >= thresholds["hit_rate_min_sample"] and hit_rate_pct is not None:
            hit_rate_summary = f"Hit-rate: {hit_rate_pct:.0f}% ({hit_rate_result['hits']}/{total} directional decisions correct)"
        else:
            hit_rate_summary = f"Hit-rate: insufficient data ({total} directional decisions so far)"
        enriched_memory = hit_rate_result["enriched_memory"]

    n_bullish = sum(1 for s in analyst_signals if str(s.signal).lower() == "bullish")
    n_bearish = sum(1 for s in analyst_signals if str(s.signal).lower() == "bearish")
    signal_balance = n_bullish - n_bearish
    if enriched_memory:
        prompt = MARKET_TIMING_PROMPT_W_MEMORY.format(
            ticker=ticker,
            analyst_signals=analyst_signals,
            signal_balance=f"{signal_balance:+d} ({n_bullish} bullish, {n_bearish} bearish)",
            decision_memory=enriched_memory,
            hit_rate_summary=hit_rate_summary,
            current_price=current_price,
        )
    else:
        prompt = MARKET_TIMING_PROMPT.format(
            ticker=ticker,
            analyst_signals=analyst_signals,
            signal_balance=f"{signal_balance:+d} ({n_bullish} bullish, {n_bearish} bearish)",
            decision_memory=decision_memory,
            current_price=current_price,
        )

    ticker_decision = agent_call(prompt=prompt, llm_config=llm_config, pydantic_model=Decision)
    ticker_decision.price = current_price
    ticker_decision.shares = 1 if ticker_decision.action.value in ("Buy", "Sell") else 0

    logger.log_agent_status(
        AgentKey.PORTFOLIO_ENRICHED_MEMORY if enriched_memory else AgentKey.PORTFOLIO, ticker, "Market timing decision made"
    )
    logger.log_decision(ticker, ticker_decision)
    return {"decision": ticker_decision}


def portfolio_agent_enriched_memory(state: FundState):
    """Portfolio manager with decision memory."""
    return _run_portfolio(state, enriched_memory=True)


def portfolio_agent(state: FundState):
    """Portfolio manager without decision memory — signals only."""
    return _run_portfolio(state, enriched_memory=False)


def calculate_ticker_shares(portfolio, current_price, ticker, optimal_position_ratio):
    """Calculate current and tradable shares for a ticker based on NAV and optimal position ratio."""
    current_shares = portfolio.positions[ticker].shares if ticker in portfolio.positions else 0
    current_value = current_shares * current_price
    total_portfolio_value = portfolio.cashflow + sum(portfolio.positions[t].value for t in portfolio.positions)
    position_limit = total_portfolio_value * optimal_position_ratio
    position_value_gap = position_limit - current_value
    if position_value_gap > 0:
        tradable_shares = min(position_value_gap, portfolio.cashflow) // current_price
    else:
        tradable_shares = max(position_value_gap // current_price, -current_shares)
    logger.info(
        f"Ticker {ticker}: current_shares={current_shares}, current_price={current_price}, "
        f"position_limit=${position_limit:.2f}, position_value_gap=${position_value_gap:.2f}, "
        f"tradable_shares={tradable_shares}"
    )
    return current_shares, tradable_shares


def portfolio_agent_risk_managed(state: FundState):
    """Risk-managed portfolio agent: risk control LLM call → NAV-based share sizing → decision LLM call."""
    agent_name = AgentKey.PORTFOLIO_RISK_MANAGED
    portfolio = state["portfolio"]
    ticker = state["ticker"]
    exp_name = state["exp_name"]
    trading_date = state["trading_date"]
    analyst_signals = state["analyst_signals"]
    llm_config = state["llm_config"]
    num_tickers = state["num_tickers"]
    api_payload = state.get("api_payload")
    market_date = trading_date if api_payload else trading_date - datetime.timedelta(days=1)

    db = get_db()

    try:
        current_price = load_specific_data(
            symbol=ticker,
            date=market_date,
            type="current_price",
            api_payload=api_payload,
        )
    except Exception as e:
        logger.error(f"Failed to fetch price data for {ticker}: {e}")
        raise RuntimeError("Failed to make decision") from e

    max_position_ratio = 1
    if num_tickers > 1:
        max_position_ratio = round(2 / num_tickers * 20) / 20

    risk_prompt = RISK_CONTROL_PROMPT.format(
        ticker_signals=analyst_signals,
        portfolio=portfolio.model_dump_json(),
        max_position_ratio=max_position_ratio,
    )
    position_risk = agent_call(prompt=risk_prompt, llm_config=llm_config, pydantic_model=PositionRisk)
    logger.log_agent_status(agent_name, ticker, "Risk control")

    position_risk.optimal_position_ratio = max(0, min(position_risk.optimal_position_ratio, max_position_ratio))

    decision_memory = db.get_decision_memory(exp_name, ticker, thresholds["decision_memory_limit"])
    current_shares, tradable_shares = calculate_ticker_shares(
        portfolio, current_price, ticker, position_risk.optimal_position_ratio
    )

    prompt = PORTFOLIO_PROMPT.format(
        decision_memory=decision_memory,
        current_price=current_price,
        current_shares=current_shares,
        tradable_shares=tradable_shares,
    )
    ticker_decision = agent_call(prompt=prompt, llm_config=llm_config, pydantic_model=Decision)
    ticker_decision.price = current_price
    if ticker_decision.shares < 0 and ticker_decision.action == Action.SELL:
        ticker_decision.shares = -ticker_decision.shares

    logger.log_agent_status(agent_name, ticker, "Decision made")
    logger.log_decision(ticker, ticker_decision)
    return {"decision": ticker_decision}
