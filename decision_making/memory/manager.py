"""Layered memory manager for multi-agent sequential decision-making.

Implements a two-tier memory hierarchy inspired by InvestorBench:
  - Working memory  : short-lived context, fast decay (3-day recency half-life)
  - Long-term memory: consolidated patterns, slow decay (90-day recency half-life)

Memories flow upward (working → long-term) when their importance crosses a
consolidation threshold, and back downward when long-term importance falls
below a demotion threshold.  Each maintenance cycle applies multiplicative
importance decay and exponential recency decay, then removes entries that
drop below minimum viability thresholds.

Retrieval uses a compound score:
    score = TF-IDF cosine similarity + normalised importance + recency
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from memory.schema import LAYER_CONFIG, MemoryEntry, MemoryLayer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from database.interface import BaseDB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tfidf_similarities(query: str, texts: list[str]) -> list[float]:
    """Return cosine similarities between query and each text via TF-IDF."""
    if not texts:
        return []
    corpus = [*texts, query]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vec = tfidf_matrix[-1]
        doc_vecs = tfidf_matrix[:-1]
        sims = cosine_similarity(query_vec, doc_vecs)[0]
        return sims.tolist()
    except Exception:
        return [0.0] * len(texts)


def _format_memory_block(memories: list[MemoryEntry], layer_label: str) -> str:
    """Format a list of memories into a readable block for LLM prompts."""
    if not memories:
        return f"[{layer_label}: no relevant memories]"
    lines = [f"[{layer_label}]"]
    for i, m in enumerate(memories, 1):
        lines.append(f"  {i}. [{m.created_date}] (importance={m.importance:.1f}) {m.text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------


class MemoryManager:
    """Manages working memory and long-term memory for a single experiment.

    All persistence is delegated to the BaseDB implementation (SQLite in
    production) so this class contains only computation logic.
    """

    def __init__(self, db: BaseDB) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add_to_working_memory(self, exp_name: str, ticker: str, created_date: str, text: str) -> int | None:
        """Ingest a new observation into working memory."""
        cfg = LAYER_CONFIG[MemoryLayer.WORKING]
        entry = MemoryEntry(
            exp_name=exp_name,
            ticker=ticker,
            created_date=created_date,
            text=text,
            importance=cfg["init_importance"],
            recency=1.0,
            delta=0,
            layer=MemoryLayer.WORKING,
        )
        mem_id = self.db.save_memory(entry)
        logger.debug("Working memory added for %s/%s: %s", exp_name, ticker, text[:60])
        return mem_id

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query_memories(
        self,
        exp_name: str,
        ticker: str,
        query_text: str,
        top_k: int = 6,
    ) -> tuple[list[MemoryEntry], list[MemoryEntry]]:
        """Retrieve the most relevant memories from both layers.

        Returns:
            (working_memories, long_term_memories) - each sorted by
            compound score descending, capped at top_k total.
        """
        all_memories = self.db.get_memories(exp_name, ticker)
        if not all_memories:
            return [], []

        texts = [m.text for m in all_memories]
        similarities = _tfidf_similarities(query_text, texts)

        scored = sorted(
            zip(similarities, all_memories),
            key=lambda pair: pair[1].compound_score(pair[0]),
            reverse=True,
        )

        top = [mem for _, mem in scored[:top_k]]
        working = [m for m in top if m.layer == MemoryLayer.WORKING]
        long_term = [m for m in top if m.layer == MemoryLayer.LONG_TERM]
        return working, long_term

    def format_memory_context(self, working: list[MemoryEntry], long_term: list[MemoryEntry]) -> str:
        """Render memory entries as a prompt-ready string."""
        parts = [
            _format_memory_block(working, "Working Memory"),
            _format_memory_block(long_term, "Long-Term Memory"),
        ]
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def decay_and_maintain(self, exp_name: str) -> None:
        """Apply one maintenance cycle: decay → layer transitions → cleanup.

        Should be called once per trading day, after decisions are recorded.
        """
        memories = self.db.get_all_memories(exp_name)
        updates: list[MemoryEntry] = []
        deletions: list[int] = []

        for mem in memories:
            cfg = LAYER_CONFIG[mem.layer]
            new_delta = mem.delta + 1
            new_importance = mem.importance * cfg["decay_rate"]
            new_recency = MemoryEntry.compute_recency(new_delta, cfg["recency_factor"])

            # Cleanup: remove entries that fall below viability thresholds
            if new_importance < cfg["importance_cleanup"] or new_recency < cfg["recency_cleanup"]:
                if mem.id is not None:
                    deletions.append(mem.id)
                continue

            # Layer transitions
            new_layer = mem.layer
            reset = False

            if mem.layer == MemoryLayer.WORKING and new_importance >= cfg["consolidation_threshold"]:
                # Promote to long-term
                new_layer = MemoryLayer.LONG_TERM
                new_importance = LAYER_CONFIG[MemoryLayer.LONG_TERM]["init_importance"]
                reset = True
                logger.debug("Memory %s promoted to long-term (ticker=%s)", mem.id, mem.ticker)

            elif mem.layer == MemoryLayer.LONG_TERM:
                lt_cfg = LAYER_CONFIG[MemoryLayer.LONG_TERM]
                if new_importance < lt_cfg["demotion_threshold"]:
                    new_layer = MemoryLayer.WORKING
                    logger.debug("Memory %s demoted to working (ticker=%s)", mem.id, mem.ticker)

            if reset:
                new_recency = 1.0
                new_delta = 0

            updates.append(
                MemoryEntry(
                    id=mem.id,
                    exp_name=mem.exp_name,
                    ticker=mem.ticker,
                    created_date=mem.created_date,
                    text=mem.text,
                    importance=new_importance,
                    recency=new_recency,
                    delta=new_delta,
                    layer=new_layer,
                    access_count=mem.access_count,
                )
            )

        if updates:
            self.db.update_memories(updates)
        if deletions:
            self.db.delete_memories(deletions)

        logger.info(
            "Memory maintenance: %d updated, %d deleted for %s",
            len(updates),
            len(deletions),
            exp_name,
        )

    def update_importance_from_feedback(
        self,
        memory_ids: list[int],
        positive: bool,
        step: float = 5.0,
    ) -> None:
        """Reinforce or penalise memories that contributed to a trade outcome.

        Positive feedback (profitable trade) increases importance; negative
        feedback (losing trade) decreases it.  Importance is clamped to [0, 100].
        """
        if not memory_ids:
            return
        memories = self.db.get_memories_by_ids(memory_ids)
        delta = step if positive else -step
        updated = []
        for mem in memories:
            mem.importance = max(0.0, min(100.0, mem.importance + delta))
            mem.access_count += 1
            updated.append(mem)
        if updated:
            self.db.update_memories(updated)
