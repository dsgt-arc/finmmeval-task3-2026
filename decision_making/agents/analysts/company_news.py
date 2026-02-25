import datetime

from graph.constants import AgentKey
from graph.schema import AnalystSignal, FundState
from llm.cost_estimation import estimate_cost
from llm.inference import agent_call
from llm.prompt import COMPANY_NEWS_PROMPT
from util.db_helper import get_db
from util.logger import logger

from decision_making.data import load_specific_data

# thresholds
thresholds = {
    "news_count": 10,
}


def company_news_agent(state: FundState):
    """News specialist analyzing company news to provide a signal."""
    agent_name = AgentKey.COMPANY_NEWS
    ticker = state["ticker"]
    trading_date = state["trading_date"]
    llm_config = state["llm_config"]
    portfolio_id = state["portfolio"].id

    # Get db instance
    db = get_db()

    logger.log_agent_status(agent_name, ticker, "Fetching company news")

    # Get the company news
    try:
        # access t-1 news data for ticker
        historic_date = trading_date - datetime.timedelta(days=1)
        company_news = load_specific_data(symbol=ticker, date=historic_date, type="news")
    except Exception as e:
        logger.error(f"Failed to fetch company news for {ticker}: {e}")
        return state

    # Analyze news sentiment via LLM
    prompt = COMPANY_NEWS_PROMPT.format(news=company_news)

    # Estimate and log cost before making the call
    costs = estimate_cost(prompt, llm_config)
    logger.info(f"[{agent_name}] Estimated cost for {ticker}: ${costs:.6f}")

    # Get LLM signal
    signal = agent_call(
        prompt=prompt,
        llm_config=llm_config,
        pydantic_model=AnalystSignal,
    )

    # save signal
    logger.log_signal(agent_name, ticker, signal)
    db.save_signal(portfolio_id, agent_name, ticker, prompt, signal)

    return {"analyst_signals": [signal]}
