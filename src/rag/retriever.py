"""
Investigation retriever for the Flame in Freefall RAG subsystem.

Provides two-stage retrieval over document chunks:
  1. Fast candidate selection via TF-IDF cosine similarity.
  2. Re-ranking with BM25-style term scoring + metadata boosts.

Supports keyword search, semantic search (via an optional embeddings
manager), and hybrid search that linearly blends both scores.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np
from src.catalog.compat_pydantic import BaseModel, Field
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    from src.ai.compat_ml import TfidfVectorizer, cosine_similarity

from src.rag.chunker import DocumentChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class RetrievalResult(BaseModel):
    """A single retrieval hit, wrapping a :class:`DocumentChunk`."""

    chunk: DocumentChunk
    score: float = Field(..., description="Relevance score (0–1 normalised).")
    match_type: str = Field(
        ...,
        description="How the chunk was matched: 'keyword', 'semantic', or 'hybrid'.",
    )
    highlighted_text: str = Field(
        ...,
        description="Chunk text with matching query terms wrapped in **bold**.",
    )


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class InvestigationRetriever:
    """
    Retrieves relevant document chunks for a given query.

    Two-stage retrieval:
      1. **Stage 1 – TF-IDF**: Vectorise all chunks and compute cosine
         similarity against the query.  Return the top *candidate_pool*
         candidates.
      2. **Stage 2 – Re-rank**: Score each candidate with a lightweight
         BM25-inspired formula plus metadata boosts (title match,
         description chunks rank higher than context, etc.).

    An optional ``embeddings_manager`` (expected to expose
    ``encode(texts) -> np.ndarray``) enables semantic search and
    hybrid (keyword + semantic) retrieval.
    """

    # BM25 parameters
    _K1: float = 1.5
    _B: float = 0.75

    def __init__(self, embeddings_manager: Any | None = None) -> None:
        self.chunks: list[DocumentChunk] = []
        self.embeddings_manager = embeddings_manager

        # TF-IDF index
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix: np.ndarray | None = None

        # Inverted index for BM25 re-ranking
        self._inv_index: dict[str, list[tuple[int, float]]] = {}
        self._doc_lens: list[int] = []
        self._avg_dl: float = 0.0

        # Semantic embeddings (populated when embeddings_manager is present)
        self._embeddings: np.ndarray | None = None

        # Lookup helpers
        self._id_to_indices: dict[str, list[int]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[DocumentChunk]) -> None:
        """Build search indices from a list of document chunks.

        Constructs:
          * A TF-IDF sparse matrix for fast cosine retrieval.
          * An inverted index for BM25-style re-ranking.
          * (If *embeddings_manager* available) dense semantic embeddings.
          * An investigation-id → chunk-index lookup table.
        """
        if not chunks:
            logger.warning("build_index called with zero chunks – nothing to index.")
            return

        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]

        # --- TF-IDF ---
        self._vectorizer = TfidfVectorizer(
            max_features=20_000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)

        # --- Inverted index for BM25 ---
        self._build_bm25_index(texts)

        # --- ID lookup ---
        self._id_to_indices.clear()
        for idx, chunk in enumerate(self.chunks):
            self._id_to_indices[chunk.investigation_id].append(idx)

        # --- Semantic embeddings ---
        if self.embeddings_manager is not None:
            try:
                self._embeddings = self.embeddings_manager.encode(texts)
                logger.info("Computed semantic embeddings for %d chunks.", len(texts))
            except Exception:
                logger.exception("Failed to compute semantic embeddings.")
                self._embeddings = None
        else:
            self._embeddings = None

        logger.info("Built index over %d chunks.", len(self.chunks))

    # ------------------------------------------------------------------
    # Public retrieval API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the *top_k* most relevant chunks for *query*.

        Parameters
        ----------
        query : str
            Free-text search query.
        top_k : int
            Number of results to return.
        filters : dict, optional
            Key-value pairs matched against chunk *metadata* (e.g.
            ``{"category": "combustion"}``).  All supplied filters must
            match (AND logic).

        Returns
        -------
        list[RetrievalResult]
            Descending-score list capped at *top_k*.
        """
        self._require_index()

        if not query.strip():
            return []

        # Stage 1: TF-IDF cosine similarity
        candidate_scores = self._tfidf_retrieve(query)

        # Stage 2: BM25 re-rank
        reranked = self._bm25_rerank(query, candidate_scores)

        # Apply filters
        if filters:
            reranked = self._apply_filters(reranked, filters)

        # Normalise scores to 0-1
        reranked = self._normalise_scores(reranked)

        # Build results
        results = self._build_results(query, reranked[:top_k], match_type="keyword")
        return results

    def retrieve_by_investigation(self, investigation_id: str) -> list[DocumentChunk]:
        """Return all indexed chunks belonging to *investigation_id*."""
        indices = self._id_to_indices.get(investigation_id, [])
        return [self.chunks[i] for i in indices]

    def hybrid_search(
        self,
        query: str,
        keyword_weight: float = 0.4,
        semantic_weight: float = 0.6,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Combine keyword (TF-IDF + BM25) and semantic search.

        When no embeddings manager is configured the method falls back to
        pure keyword retrieval (with ``keyword_weight = 1.0``).

        Parameters
        ----------
        query : str
            Free-text query.
        keyword_weight : float
            Weight for keyword scores (default 0.4).
        semantic_weight : float
            Weight for semantic scores (default 0.6).
        top_k : int
            Maximum results to return.
        filters : dict, optional
            Metadata filters (AND logic).

        Returns
        -------
        list[RetrievalResult]
        """
        self._require_index()

        if not query.strip():
            return []

        # --- Keyword component ---
        kw_scores = self._tfidf_retrieve(query)
        kw_scores = self._bm25_rerank(query, kw_scores)
        kw_scores = dict(self._normalise_scores(list(kw_scores.items())))

        # --- Semantic component ---
        sem_scores: dict[int, float] = {}
        if self._embeddings is not None and self.embeddings_manager is not None:
            try:
                q_emb = self.embeddings_manager.encode([query])
                sims = cosine_similarity(q_emb, self._embeddings)[0]
                for idx, sim in enumerate(sims):
                    sem_scores[idx] = float(max(sim, 0.0))
                # Normalise
                sem_scores = dict(self._normalise_scores(list(sem_scores.items())))
            except Exception:
                logger.exception("Semantic scoring failed – falling back to keyword-only.")

        if not sem_scores:
            keyword_weight = 1.0
            semantic_weight = 0.0

        # --- Blend ---
        all_indices = set(kw_scores) | set(sem_scores)
        combined: dict[int, float] = {}
        for idx in all_indices:
            kw = kw_scores.get(idx, 0.0)
            se = sem_scores.get(idx, 0.0)
            combined[idx] = keyword_weight * kw + semantic_weight * se

        # Sort descending
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        # Apply filters
        if filters:
            ranked = self._apply_filters(dict(ranked), filters)
            ranked = sorted(ranked.items(), key=lambda x: x[1], reverse=True)

        match_type = "hybrid" if sem_scores else "keyword"
        results = self._build_results(query, ranked[:top_k], match_type=match_type)
        return results

    # ------------------------------------------------------------------
    # Internal – TF-IDF retrieval
    # ------------------------------------------------------------------

    def _tfidf_retrieve(self, query: str) -> dict[int, float]:
        """Return {chunk_index: cosine_score} for all chunks."""
        assert self._vectorizer is not None and self._tfidf_matrix is not None
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
        return {idx: float(s) for idx, s in enumerate(scores) if s > 0}

    # ------------------------------------------------------------------
    # Internal – BM25 re-ranking
    # ------------------------------------------------------------------

    def _build_bm25_index(self, texts: list[str]) -> None:
        """Construct inverted index and document-length stats for BM25."""
        self._inv_index.clear()
        self._doc_lens = []

        for idx, text in enumerate(texts):
            tokens = self._tokenise(text)
            self._doc_lens.append(len(tokens))
            tf: Counter[str] = Counter(tokens)
            for term, freq in tf.items():
                self._inv_index.setdefault(term, []).append((idx, freq))

        total_words = sum(self._doc_lens) if self._doc_lens else 1
        self._avg_dl = total_words / max(len(self._doc_lens), 1)

    def _bm25_rerank(
        self, query: str, candidate_scores: dict[int, float]
    ) -> dict[int, float]:
        """Re-rank *candidate_scores* with BM25 + chunk-type boosts."""
        if not candidate_scores:
            return {}

        q_tokens = self._tokenise(query)
        n_docs = len(self.chunks)
        bm25_scores: dict[int, float] = defaultdict(float)

        for term in q_tokens:
            postings = self._inv_index.get(term, [])
            df = len(postings)
            if df == 0:
                continue
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for doc_idx, tf in postings:
                if doc_idx not in candidate_scores:
                    continue
                dl = self._doc_lens[doc_idx]
                numerator = tf * (self._K1 + 1)
                denominator = tf + self._K1 * (
                    1 - self._B + self._B * dl / self._avg_dl
                )
                bm25_scores[doc_idx] += idf * numerator / denominator

        # Blend TF-IDF cosine with BM25 (50/50) + chunk-type boost.
        combined: dict[int, float] = {}
        for idx in candidate_scores:
            tfidf_s = candidate_scores[idx]
            bm25_s = bm25_scores.get(idx, 0.0)
            boost = self._chunk_type_boost(self.chunks[idx])
            combined[idx] = (0.5 * tfidf_s + 0.5 * bm25_s) * boost

        return combined

    @staticmethod
    def _chunk_type_boost(chunk: DocumentChunk) -> float:
        """Give slight preference to more informative chunk types."""
        boosts: dict[str, float] = {
            "description": 1.15,
            "objective": 1.10,
            "application": 1.05,
            "keyword_summary": 1.00,
            "context": 0.95,
        }
        return boosts.get(chunk.chunk_type, 1.0)

    # ------------------------------------------------------------------
    # Internal – filtering, normalisation, result construction
    # ------------------------------------------------------------------

    def _apply_filters(
        self, scores: dict[int, float], filters: dict[str, Any]
    ) -> dict[int, float]:
        """Remove candidates whose metadata does not match *filters*."""
        filtered: dict[int, float] = {}
        for idx, score in scores.items():
            chunk = self.chunks[idx]
            if self._matches_filters(chunk, filters):
                filtered[idx] = score
        return filtered

    @staticmethod
    def _matches_filters(chunk: DocumentChunk, filters: dict[str, Any]) -> bool:
        """Check that every filter key matches the chunk metadata (AND)."""
        meta = chunk.metadata
        for key, required in filters.items():
            actual = meta.get(key)
            if actual is None:
                return False
            # Support list membership (e.g. categories filter)
            if isinstance(required, list):
                if isinstance(actual, list):
                    if not set(required) & set(actual):
                        return False
                elif actual not in required:
                    return False
            elif isinstance(actual, list):
                if required not in actual:
                    return False
            else:
                if str(actual).lower() != str(required).lower():
                    return False
        return True

    @staticmethod
    def _normalise_scores(
        items: Any,
    ) -> list[tuple[int, float]]:
        """Normalise scores to [0, 1] and sort descending."""
        if not items:
            return []
        if isinstance(items, dict):
            pair_list = list(items.items())
        else:
            pair_list = list(items)
        max_s = max(s for _, s in pair_list)
        if max_s <= 0:
            return [(idx, 0.0) for idx, _ in pair_list]
        normed = [(idx, s / max_s) for idx, s in pair_list]
        normed.sort(key=lambda x: x[1], reverse=True)
        return normed

    def _build_results(
        self,
        query: str,
        ranked: Sequence[tuple[int, float]],
        match_type: str,
    ) -> list[RetrievalResult]:
        """Wrap raw (index, score) pairs into :class:`RetrievalResult` objects."""
        q_tokens = set(self._tokenise(query))
        results: list[RetrievalResult] = []
        for idx, score in ranked:
            chunk = self.chunks[idx]
            highlighted = self._highlight(chunk.text, q_tokens)
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=round(score, 6),
                    match_type=match_type,
                    highlighted_text=highlighted,
                )
            )
        return results

    @staticmethod
    def _highlight(text: str, terms: set[str]) -> str:
        """Wrap each occurrence of any term in the text with **bold**."""
        if not terms:
            return text
        # Build a single regex alternation (case-insensitive, word-bounded).
        escaped = [re.escape(t) for t in terms if t]
        if not escaped:
            return text
        pattern = re.compile(
            r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE
        )
        return pattern.sub(r"**\1**", text)

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """Lowercase tokenisation with minimal stop-word removal."""
        _STOP = frozenset(
            {
                "a", "an", "the", "is", "are", "was", "were", "be", "been",
                "being", "have", "has", "had", "do", "does", "did", "will",
                "would", "could", "should", "may", "might", "shall", "can",
                "of", "in", "to", "for", "with", "on", "at", "from", "by",
                "about", "as", "into", "through", "during", "before",
                "after", "above", "below", "between", "and", "but", "or",
                "not", "no", "nor", "so", "yet", "both", "either", "neither",
                "each", "every", "all", "any", "this", "that", "these",
                "those", "it", "its", "i", "we", "you", "he", "she", "they",
                "me", "us", "him", "her", "them", "my", "our", "your",
                "his", "their", "what", "which", "who", "whom", "how",
                "when", "where", "why",
            }
        )
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [t for t in tokens if t not in _STOP and len(t) > 1]

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    def _require_index(self) -> None:
        """Raise if the index has not been built yet."""
        if self._tfidf_matrix is None or not self.chunks:
            raise RuntimeError(
                "No index available. Call build_index(chunks) first."
            )
