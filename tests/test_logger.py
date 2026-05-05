"""Regression tests for the DS@GT StockTron logger."""

from __future__ import annotations

from pathlib import Path
import sys
import logging

ROOT = Path(__file__).resolve().parents[1]
DECISION_MAKING_DIR = ROOT / "decision_making"
for path in (str(DECISION_MAKING_DIR), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from util.logger import logger as stocktron_logger  # noqa: E402


def test_console_logs_go_to_stderr():
    stream_handlers = [
        handler
        for handler in stocktron_logger.logger.handlers
        if type(handler) is logging.StreamHandler
    ]

    assert stream_handlers, "expected at least one console stream handler"
    assert any(handler.stream is sys.stderr for handler in stream_handlers)
