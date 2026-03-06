from graph.constants import AgentKey
from graph.schema import AnalystSignal, FundState
from util.db_helper import get_db
from util.logger import logger


def dummy_agent(state: FundState):
    """Dummy agent that does nothing - for debugging."""
    signal = AnalystSignal(signal="Bullish", justification="Dummy agent - no analysis")
    logger.log_signal(AgentKey.DUMMY, state["ticker"], signal)

    # Get db instance
    db = get_db()
    db.save_signal(state["portfolio"].id, AgentKey.DUMMY, state["ticker"], "DUMMY", signal)

    return {"analyst_signals": [signal]}
