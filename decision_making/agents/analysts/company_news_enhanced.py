import datetime

from decision_making.ama_data import load_specific_data
from graph.constants import AgentKey, Signal
from graph.schema import AnalystSignal, FundState, RelevanceCheck
from llm.cost_estimation import estimate_cost
from llm.inference import agent_call
from llm.prompt import COMPANY_NEWS_ENHANCED_PROMPT, RELEVANCE_CHECK_PROMPT
from util.db_helper import get_db
from util.logger import logger

# Configuration
CONFIG = {
    "lookback_days": 10,
    "min_history_for_trend": 2,
    "trend_threshold": 0.15,
}


def _average_sentiment(article_sentiments: list[AnalystSignal]) -> AnalystSignal:
    """Majority vote of article sentiments to get overall signal."""

    # count AnalystSignal
    counts = {Signal.BULLISH: 0, Signal.BEARISH: 0, Signal.NEUTRAL: 0}
    strengths = {Signal.BULLISH: [], Signal.BEARISH: [], Signal.NEUTRAL: []}
    for s in article_sentiments:
        counts[s.signal] += 1
        strengths[s.signal].append(s.signal_strength)

    # if there's a clear majority, return it
    signal = max(counts, key=counts.get)
    justification = f"Based on {len(article_sentiments)} articles: " + ", ".join([
        f"{s.signal} (strength {s.signal_strength})" for s in article_sentiments if s.signal == signal
    ])
    valid_strengths = [v for v in strengths[signal] if v is not None]
    strength = sum(valid_strengths) / len(valid_strengths) if valid_strengths else 0.0

    if not (counts[signal] > len(article_sentiments) / 2):
        # no clear majority, return NEUTRAL
        signal = Signal.NEUTRAL
        justification = f"No clear majority found among {len(article_sentiments)} articles."
        strength = 0.0

    return AnalystSignal(signal=signal, justification=justification, signal_strength=strength)


def calculate_sentiment_trend_change(
    historical_signals: list[dict], current_signal: AnalystSignal, trend_threshold: float = 0.15
) -> tuple[str, float]:
    """Calculate trend and change from historical signals."""

    # Extract strengths (oldest first for trend calculation)
    strengths = [s["signal_strength"] for s in reversed(historical_signals)]

    # Calculate change from most recent
    sentiment_change = current_signal.signal_strength - strengths[-1]
    normalized_change = abs(sentiment_change) / 2  # normalized by two as change can range from -2 to +2
    if sentiment_change > 0:
        sentiment_change_signal = AnalystSignal(
            signal=Signal.BULLISH,
            justification=f"Sentiment increased from day before by {sentiment_change:.2f}",
            signal_strength=normalized_change,
        )
    elif sentiment_change < 0:
        sentiment_change_signal = AnalystSignal(
            signal=Signal.BEARISH,
            justification=f"Sentiment decreased from day before by {abs(sentiment_change):.2f}",
            signal_strength=normalized_change,
        )
    else:
        sentiment_change_signal = AnalystSignal(
            signal=Signal.NEUTRAL, justification="No change in sentiment from day before", signal_strength=0.0
        )

    # Calculate trend: compare first half vs second half averages
    mid = len(strengths) // 2
    older_avg = sum(strengths[:mid]) / mid
    recent_avg = sum(strengths[mid:]) / (len(strengths) - mid)
    trend_slope = recent_avg - older_avg
    normalized_slope = trend_slope / (max(abs(older_avg), abs(recent_avg)) + 1e-6)  # avoid div by zero
    normalized_slope = max(-1.0, min(1.0, normalized_slope))  # clamp to [-1, 1]

    # Classify trend
    if trend_slope > trend_threshold:
        trend_signal = AnalystSignal(
            signal=Signal.BULLISH,
            justification=f"Sentiment trend is increasing with slope {trend_slope:.2f}",
            signal_strength=normalized_slope,
        )
    elif trend_slope < -trend_threshold:
        trend_signal = AnalystSignal(
            signal=Signal.BEARISH,
            justification=f"Sentiment trend is decreasing with slope {trend_slope:.2f}",
            signal_strength=normalized_slope,
        )
    else:
        trend_signal = AnalystSignal(
            signal=Signal.NEUTRAL,
            justification=f"Sentiment trend is stable with slope {trend_slope:.2f}",
            signal_strength=normalized_slope,
        )

    return trend_signal, sentiment_change_signal


def _split_by_headlines(text: str) -> dict[str, list[str]]:
    lines = text.split("\n")

    sections = {}
    current_headline = "intro"
    sections[current_headline] = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Detect headline (no dash, no colon at end, not bullet)
        if not line.startswith("-") and not line.endswith(":") and len(line.split()) <= 12:
            current_headline = line.lower()
            sections[current_headline] = []
            continue

        sections[current_headline].append(line)

    return sections


def _is_relevant_for_trading(ticker: str, topic: str, content: list[str], agent_name: str, llm_config: dict) -> bool:
    """Use LLM to filter out sections unlikely to impact next-day prices."""

    # Skip very short sections
    if len(content.split()) < 10:
        return False

    try:
        prompt = RELEVANCE_CHECK_PROMPT.format(ticker=ticker, topic=topic, content=content)
        # Estimate and log cost before making the call
        costs = estimate_cost(prompt, llm_config)
        logger.info(f"[{agent_name}] Relevance Check - Estimated cost for {ticker}: ${costs:.6f}")

        relevance = agent_call(prompt=prompt, llm_config=llm_config, pydantic_model=RelevanceCheck)
        logger.debug(f" Relevance Check - {topic}: is_relevant={relevance.is_relevant}")

        return relevance.is_relevant
    except Exception:
        # On error, include the section (fail open)
        return True


def company_news_enhanced_agent(state: FundState):
    """Enhanced news sentiment analyzer with per-article analysis and trend tracking."""
    agent_name = AgentKey.COMPANY_NEWS_ENHANCED
    ticker = state["ticker"]
    trading_date = state["trading_date"]
    llm_config = state["llm_config"]
    portfolio_id = state["portfolio_id"]
    exp_name = state["exp_name"]
    db = get_db()

    # Load news data
    try:
        historic_date = trading_date - datetime.timedelta(days=1)
        news_data = load_specific_data(symbol=ticker, date=historic_date, type="news")
        news_data = (
            news_data if isinstance(news_data, str) else "".join(news_data)
        )  # data error in history, a few days with 2 news articles

        if not news_data:
            signal = AnalystSignal(signal=Signal.NEUTRAL, justification="No news available", signal_strength=0.0)
            db.save_signal(portfolio_id, agent_name, ticker, "No news", signal)
            return {"analyst_signals": [signal]}
    except Exception as e:
        logger.error(f"Failed to fetch news for {ticker}: {e}")
        return {"analyst_signals": []}

    # Analyze article in different dimensions
    logger.log_agent_status(agent_name, ticker, "Estimating enhanceed sentiment via relevance-based and pargraph-level analysis")
    parts_of_news = _split_by_headlines(news_data)
    article_sentiments = []
    for topic, part in parts_of_news.items():
        # Filter: skip sections not relevant for price predictions using LLM
        if not _is_relevant_for_trading(ticker, topic, " ".join(part), agent_name, llm_config):
            continue

        try:
            prompt = COMPANY_NEWS_ENHANCED_PROMPT.format(ticker=ticker, topic=topic, content=" ".join(part))

            # Estimate and log cost before making the call
            costs = estimate_cost(prompt, llm_config)
            logger.info(f"[{agent_name}] Sentiment Analysis - Estimated cost for {ticker}: ${costs:.6f}")

            sentiment = agent_call(prompt=prompt, llm_config=llm_config, pydantic_model=AnalystSignal)
            article_sentiments.append(sentiment)
        except Exception as e:
            logger.warning(f"Article analysis failed: {e}")
            continue

    if not article_sentiments:
        signal = AnalystSignal(
            signal=Signal.NEUTRAL,
            justification="Failed to analyze articles",
            signal_strength=0.0,
        )
        db.save_signal(portfolio_id, agent_name, ticker, "Analysis failed", signal)
        return {"analyst_signals": [signal]}

    # Aggregate sentiments
    logger.log_agent_status(agent_name, ticker, "Estimating average sentiment signal from paragraph-level analysis")
    sentiment_signal = _average_sentiment(article_sentiments)

    # Get historical signals and calculate trend
    logger.log_agent_status(agent_name, ticker, "Estimating sentiment change and trend based on historical sentiment signals")
    historical_signals = db.get_signal_history(exp_name, ticker, agent_name + "_sentiment", CONFIG["lookback_days"])
    if len(historical_signals) >= CONFIG["min_history_for_trend"]:
        sentiment_trend, sentiment_change = calculate_sentiment_trend_change(
            historical_signals, sentiment_signal, CONFIG["trend_threshold"]
        )
    else:
        sentiment_trend = AnalystSignal(
            signal=Signal.NEUTRAL, justification="Not enough history for sentiment trend analysis", signal_strength=0.0
        )
        sentiment_change = AnalystSignal(
            signal=Signal.NEUTRAL, justification="Not enough history for sentiment change analysis", signal_strength=0.0
        )

    # Log all sentiment signals
    # 1. Advanced sentiment signal
    logger.log_signal(agent_name + "_sentiment", ticker, sentiment_signal)
    db.save_signal(portfolio_id, agent_name + "_sentiment", ticker, prompt, sentiment_signal)

    # 2. Sentiment Trend
    logger.log_signal(agent_name + "_trend", ticker, sentiment_trend)
    db.save_signal(portfolio_id, agent_name + "_trend", ticker, "no prompt", sentiment_trend)

    # 3. Sentiment Change
    logger.log_signal(agent_name + "_change", ticker, sentiment_change)
    db.save_signal(portfolio_id, agent_name, ticker, "no prompt", sentiment_change)
    return {"analyst_signals": [sentiment_signal, sentiment_trend, sentiment_change]}
