from datetime import datetime
import logging
import os
import pathlib
import sys

from graph.schema import AnalystSignal, Decision, Portfolio


class DeepFundLogger:
    """Logger for the Deep Fund application."""

    def __init__(self, log_level: str = "INFO"):
        """Initialize the logger.

        Args:
            log_dict: Dictionary containing log configuration.
        """
        self.log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
        self.log_level = log_level

        # Create log directory if it doesn't exist
        pathlib.Path(self.log_dir).mkdir(exist_ok=True, parents=True)

        # Create logger
        self.logger = logging.getLogger("deep_fund")
        self.logger.setLevel(self.log_level)
        self.logger.propagate = False

        if self.logger.handlers:
            return

        # Create file handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"deepfund_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.log_level)

        # Send console logs to stderr so subprocess JSON written to stdout
        # stays clean for the bridge to parse.
        console_handler = logging.StreamHandler(stream=sys.stderr)
        console_handler.setLevel(self.log_level)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def debug(self, message: str, *args, **kwargs):
        """Log a debug message."""
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Log an info message."""
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Log a warning message."""
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Log an error message."""
        self.logger.error(message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs):
        """Log an error message with exception information."""
        self.logger.exception(message, *args, **kwargs)

    def log_agent_status(self, agent_name: str, ticker: str, status: str):
        """Log the status of an agent."""
        if ticker:
            msg = f"Agent: {agent_name} | Ticker: {ticker} | Status: {status}"
        else:
            msg = f"Agent: {agent_name} | Status: {status}"

        self.info(msg)

    def log_decision(self, ticker: str, d: Decision):
        """Log the decision of a ticker."""
        msg = f"Decision for {ticker}: {d.action} | Shares: {d.shares} | Price: {d.price} | Justification: {d.justification}"
        self.info(msg)

    def log_signal(self, agent_name: str, ticker: str, s: AnalystSignal):
        """Log the signal of a ticker."""
        msg = f"Agent: {agent_name} | Ticker: {ticker} | Signal: {s.signal} | Justification: {s.justification}"
        self.info(msg)

    def log_portfolio(self, label: str, p: Portfolio):
        """Log portfolio state (risk_managed mode)."""
        positions_summary = {t: {"shares": pos.shares, "value": pos.value} for t, pos in p.positions.items()}
        msg = f"{label} | ID: {p.id} | Cashflow: {p.cashflow:.2f} | Positions: {positions_summary}"
        self.info(msg)


# Create a global logger instance
logger = DeepFundLogger()
