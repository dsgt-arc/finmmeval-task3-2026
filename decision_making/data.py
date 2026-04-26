"""Backward-compatible data loader module.

This repository now keeps the data helpers in `decision_making.ama_data`,
but some tests, scripts, and notebooks still import `decision_making.data`.
Re-export the public API here so both import paths keep working.
"""

from decision_making.ama_data import *  # noqa: F401,F403
