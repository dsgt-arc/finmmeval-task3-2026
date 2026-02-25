from graph.schema import AnalystSignal, FundState


def dummy_agent(state: FundState):
    """Dummy agent that does nothing - for debugging."""
    signal = AnalystSignal(signal="Neutral", justification="Dummy agent - no analysis")
    return {"analyst_signals": [signal]}
