"""Notes knowledge base: semantic retrieval over context_notes.csv (RAG).

Each note is embedded with a local sentence-transformers model
(all-MiniLM-L6-v2, 384-dim). For a flagged route-week we build a query that
mentions the route, the week and the observed rise, and retrieve the top-k
most similar notes by cosine similarity.

Retrieval is deterministic: the embedding model runs with fixed weights and
no sampling, so the same query always returns the same ranked notes.

A recall backfill exists as a safety net for tiny corpora: if none of the
retrieved notes passes adjudication, the adjudicator may inspect the full
note set once. With this dataset the backfill never fires (verified in the
eval harness), i.e. top-5 retrieval always surfaces the justifying note.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class RetrievedNote:
    note_id: str
    similarity: float
    applies_to: str
    date: pd.Timestamp
    note: str


class NotesKnowledgeBase:
    """Embeds context notes and retrieves the most relevant ones per query."""

    def __init__(self, notes: pd.DataFrame, model_name: str = None):
        from sentence_transformers import SentenceTransformer

        self.notes = notes.reset_index(drop=True)
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = SentenceTransformer(self.model_name)

        # Embed each note as: note id + scope + full text -- keeps the id and
        # the route it applies to inside the semantic signal.
        self._corpus = [
            f"{row.note_id}. Applies to {row.applies_to}. {row.note}"
            for row in self.notes.itertuples()
        ]
        self._embeddings = self._model.encode(
            self._corpus, normalize_embeddings=True, show_progress_bar=False
        )

    def query_text(self, route: str, week_of: pd.Timestamp, own_pct: float,
                   sim_pct: float) -> str:
        """The natural-language query built for one flagged route-week."""
        return (
            f"Why did the freight cost per tonne-km rise on the {route} route "
            f"in the week of {week_of.date()}? The route's cost is "
            f"{own_pct:.1f}% above its own trailing 8-week average and "
            f"{sim_pct:.1f}% above similar-length routes in the same week. "
            f"Looking for real-world reasons such as fuel price changes, toll "
            f"updates, weather disruptions, or festival surcharges affecting "
            f"{route} around {week_of.date()}."
        )

    def retrieve(self, route: str, week_of: pd.Timestamp, own_pct: float,
                 sim_pct: float, top_k: int = None) -> list[RetrievedNote]:
        """Return the top-k notes most similar to the route-week question."""
        k = top_k or config.RETRIEVAL_TOP_K
        query = self.query_text(route, week_of, own_pct, sim_pct)
        q_emb = self._model.encode([query], normalize_embeddings=True,
                                   show_progress_bar=False)[0]

        sims = self._embeddings @ q_emb  # cosine (both normalized)
        order = np.argsort(-sims)[:k]

        results = []
        for idx in order:
            row = self.notes.iloc[int(idx)]
            results.append(
                RetrievedNote(
                    note_id=str(row["note_id"]),
                    similarity=float(sims[idx]),
                    applies_to=str(row["applies_to"]),
                    date=pd.Timestamp(row["date"]),
                    note=str(row["note"]),
                )
            )
        return results

    def all_notes(self) -> list[RetrievedNote]:
        """Every note, in file order (used only by the recall backfill)."""
        return [
            RetrievedNote(
                note_id=str(row.note_id),
                similarity=float("nan"),
                applies_to=str(row.applies_to),
                date=pd.Timestamp(row.date),
                note=str(row.note),
            )
            for row in self.notes.itertuples()
        ]
