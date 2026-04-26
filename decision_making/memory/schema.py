from dataclasses import dataclass, field
from enum import StrEnum
from math import exp


class MemoryLayer(StrEnum):
    """Memory layer tiers with distinct decay characteristics."""

    WORKING = "working"
    LONG_TERM = "long_term"


# Per-layer configuration mirrors InvestorBench decay architecture,
# simplified to two tiers as requested.
LAYER_CONFIG: dict[MemoryLayer, dict] = {
    MemoryLayer.WORKING: {
        "init_importance": 50.0,
        "recency_factor": 3.0,  # exponential half-life in days
        "decay_rate": 0.92,  # multiplicative importance loss per maintenance cycle
        "consolidation_threshold": 70.0,  # promote to long-term if importance exceeds this
        "recency_cleanup": 0.05,
        "importance_cleanup": 5.0,
    },
    MemoryLayer.LONG_TERM: {
        "init_importance": 80.0,  # reset on promotion
        "recency_factor": 90.0,  # much slower recency decay
        "decay_rate": 0.96,
        "demotion_threshold": 55.0,  # demote back to working if importance falls below this
        "recency_cleanup": 0.05,
        "importance_cleanup": 5.0,
    },
}


@dataclass
class MemoryEntry:
    """A single memory record stored in the layered memory system."""

    exp_name: str
    ticker: str
    created_date: str  # ISO date string (YYYY-MM-DD)
    text: str
    importance: float
    recency: float
    delta: int  # time steps elapsed since last decay cycle
    layer: MemoryLayer
    access_count: int = 0
    id: int | None = field(default=None)  # set after DB insertion

    def compound_score(self, similarity: float) -> float:
        """Composite retrieval score combining semantic similarity, importance, and recency."""
        normalized_importance = min(self.importance, 100.0) / 100.0
        return similarity + normalized_importance + self.recency

    @staticmethod
    def compute_recency(delta: int, recency_factor: float) -> float:
        """Exponential recency decay: exp(-delta / factor)."""
        return exp(-delta / recency_factor) if recency_factor > 0 else 0.0
