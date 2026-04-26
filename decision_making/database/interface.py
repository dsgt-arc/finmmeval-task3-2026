from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from memory.schema import MemoryEntry


class BaseDB(ABC):
    @abstractmethod
    def get_config(self, config_id: str) -> dict:
        pass

    @abstractmethod
    def get_config_id_by_name(self, exp_name: str) -> str:
        pass

    @abstractmethod
    def create_config(self, config: dict) -> str:
        pass

    @abstractmethod
    def get_latest_trading_date(self, config_id: str) -> datetime:
        pass

    @abstractmethod
    def get_latest_portfolio(self, config_id: str) -> dict:
        pass

    @abstractmethod
    def create_portfolio(self, config_id: str, cashflow: float, trading_date: datetime) -> str:
        pass

    @abstractmethod
    def copy_portfolio(self, config_id: str, portfolio: dict, trading_date: datetime) -> str:
        pass

    @abstractmethod
    def update_portfolio(self, config_id: str, portfolio: dict, trading_date: datetime) -> bool:
        pass

    @abstractmethod
    def save_decision(self, portfolio_id: str, ticker: str, prompt: str, decision: dict, trading_date: datetime) -> str:
        pass

    @abstractmethod
    def save_signal(self, portfolio_id: str, analyst: str, ticker: str, prompt: str, signal: dict) -> str:
        pass

    @abstractmethod
    def get_recent_portfolio_ids_by_config_id(self, config_id: str, limit: int) -> list:
        pass

    @abstractmethod
    def get_decision_memory(self, exp_name: str, ticker: str, limit: int) -> list:
        pass

    @abstractmethod
    def get_signal_history(self, exp_name: str, ticker: str, analyst: str, lookback_days: int = 10) -> list:
        pass

    # ------------------------------------------------------------------
    # Layered memory CRUD
    # ------------------------------------------------------------------

    @abstractmethod
    def save_memory(self, entry: MemoryEntry) -> int | None:
        """Insert a new memory entry and return its auto-generated id."""
        pass

    @abstractmethod
    def get_memories(self, exp_name: str, ticker: str) -> list[MemoryEntry]:
        """Return all memory entries for a given experiment and ticker."""
        pass

    @abstractmethod
    def get_all_memories(self, exp_name: str) -> list[MemoryEntry]:
        """Return all memory entries for a given experiment (all tickers)."""
        pass

    @abstractmethod
    def get_memories_by_ids(self, ids: list[int]) -> list[MemoryEntry]:
        """Return memory entries for the given ids."""
        pass

    @abstractmethod
    def update_memories(self, entries: list[MemoryEntry]) -> None:
        """Bulk-update importance, recency, delta, layer, and access_count."""
        pass

    @abstractmethod
    def delete_memories(self, ids: list[int]) -> None:
        """Delete memory entries by id (called during cleanup)."""
        pass
