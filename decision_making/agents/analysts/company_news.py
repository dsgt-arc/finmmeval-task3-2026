from apis.router import APISource, Router
from graph.constants import AgentKey
from graph.schema import AnalystSignal, FundState
from llm.inference import agent_call
from llm.prompt import COMPANY_NEWS_PROMPT
from util.db_helper import get_db
from util.logger import logger

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
    router = Router(APISource.ALPHA_VANTAGE)
    try:
        company_news = router.get_us_stock_news(ticker, trading_date, thresholds["news_count"])
    except Exception as e:
        logger.error(f"Failed to fetch company news for {ticker}: {e}")
        return state

    # Analyze news sentiment via LLM
    news_dict = [m.model_dump_json() for m in company_news]
    prompt = COMPANY_NEWS_PROMPT.format(news=news_dict)

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
