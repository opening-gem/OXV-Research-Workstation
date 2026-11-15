"""
Research Ranker for Flame in Freefall.

Hybrid ranking engine that scores NASA combustion investigations
by relevance to a research query using four weighted components:

1. **Keyword** — BM25-inspired term-matching score.
2. **Semantic** — Cosine similarity of embeddings.
3. **Metadata** — Category / platform / status relevance.
4. **Evidence** — Data quality and completeness.
"""

from __future__ import annotations

import math
import re
import logging
from collections import Counter
from typing import Any

import numpy as np
from src.catalog.compat_pydantic import BaseModel, Field

from src.ai.embeddings import EmbeddingsManager

logger = logging.getLogger(__name__)


# ======================================================================
# Pydantic result models
# ======================================================================


class RankedResult(BaseModel):
    """A single ranked investigation result."""

    investigation_id: str
    title: str
    overall_score: float = Field(ge=0.0, le=1.0)
    component_scores: dict[str, float]
    explanation: str
    matching_keywords: list[str] = Field(default_factory=list)
    relevant_objectives: list[str] = Field(default_factory=list)


class RankingExplanation(BaseModel):
    """Detailed explanation of why an investigation received its rank."""

    investigation_id: str
    query: str
    scores: dict[str, float]
    top_factors: list[str] = Field(default_factory=list)
    matching_text_snippets: list[str] = Field(default_factory=list)


# ======================================================================
# BM25 helpers
# ======================================================================

_STOP_WORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or "
    "such that the their then there these they this to was will with".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    """Simple lowercase tokenizer that strips stop words."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP_WORDS]


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    avg_dl: float,
    idf: dict[str, float],
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Compute BM25 score for a single document."""
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for qt in query_tokens:
        if qt not in tf:
            continue
        freq = tf[qt]
        numerator = freq * (k1 + 1)
        denominator = freq + k1 * (1 - b + b * (dl / max(avg_dl, 1)))
        score += idf.get(qt, 0.0) * numerator / denominator
    return score


# ======================================================================
# Main ranker
# ======================================================================


class ResearchRanker:
    """Ranks NASA investigations by relevance to a research query.

    Uses a hybrid scoring approach with four tunable components.

    Args:
        embeddings: A fitted :class:`EmbeddingsManager` instance.
        weights: Optional dict overriding default component weights.

    Example:
        >>> emb = EmbeddingsManager()
        >>> emb.fit([inv["text"] for inv in investigations])
        >>> ranker = ResearchRanker(emb)
        >>> results = ranker.rank("flame spread in microgravity", investigations)
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "keyword": 0.25,
        "semantic": 0.35,
        "metadata": 0.20,
        "evidence": 0.20,
    }

    # Platform keywords for metadata matching
    PLATFORM_KEYWORDS: dict[str, list[str]] = {
        "ISS": ["iss", "space station", "international space station", "orbital"],
        "Lunar": ["lunar", "moon", "artemis", "gateway"],
        "Mars": ["mars", "martian", "transit"],
        "Shuttle": ["shuttle", "sts"],
    }

    # Status keywords
    STATUS_KEYWORDS: dict[str, list[str]] = {
        "Complete": ["completed", "complete", "finished", "concluded", "results"],
        "Active": ["active", "ongoing", "current", "operating"],
        "Planned": ["planned", "upcoming", "future", "proposed"],
    }

    def __init__(
        self,
        embeddings: EmbeddingsManager,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights is not None:
            self.weights.update(weights)

        # Normalise weights so they sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(
        self,
        query: str,
        investigations: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[RankedResult]:
        """Rank investigations by relevance to *query*.

        Each investigation dict is expected to have at least:
            - ``investigation_id`` (str)
            - ``title`` (str)
            - ``text`` (str): description / objectives text.

        Optional keys used if present:
            - ``categories`` (list[str])
            - ``platform`` (str)
            - ``status`` (str)
            - ``data_completeness`` (float, 0-1)
            - ``version_count`` (int)
            - ``objectives`` (list[str])

        Args:
            query: Free-text research query.
            investigations: List of investigation dicts.
            top_k: Maximum results to return.

        Returns:
            List of :class:`RankedResult` sorted by descending
            overall score.
        """
        if not investigations:
            return []

        # Pre-compute corpus-level stats for BM25
        def _get_text(item):
            if isinstance(item, dict):
                return item.get("text") or (item.get("description", "") + " " + " ".join(item.get("objectives", [])) + " " + " ".join(item.get("keywords", [])))
            elif hasattr(item, "to_search_text"):
                return item.to_search_text()
            return str(item)

        def _get_id(item, idx):
            if isinstance(item, dict):
                return item.get("investigation_id") or item.get("id") or f"inv_{idx}"
            elif hasattr(item, "id"):
                return item.id
            return f"inv_{idx}"

        def _get_title(item):
            if isinstance(item, dict):
                return item.get("title", "Untitled")
            elif hasattr(item, "title"):
                return item.title
            return "Untitled"

        docs_tokens = [_tokenize(_get_text(inv)) for inv in investigations]
        avg_dl = float(np.mean([len(d) for d in docs_tokens])) if docs_tokens else 1.0
        n_docs = len(investigations)

        # IDF for query terms
        query_tokens = _tokenize(query)
        doc_freq: Counter[str] = Counter()
        for dt in docs_tokens:
            seen = set(dt)
            for qt in query_tokens:
                if qt in seen:
                    doc_freq[qt] += 1

        idf: dict[str, float] = {}
        for qt in query_tokens:
            df = doc_freq.get(qt, 0)
            idf[qt] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

        results: list[RankedResult] = []

        for idx, inv in enumerate(investigations):
            text_val = _get_text(inv)
            inv_id = _get_id(inv, idx)
            title_val = _get_title(inv)
            kw_score = self.keyword_score(
                query, text_val, query_tokens, docs_tokens[idx], avg_dl, idf
            )
            sem_score = self.semantic_score(query, text_val)
            meta_score = self.metadata_score(query, inv)
            ev_score = self.evidence_score(inv)

            component_scores = {
                "keyword": kw_score,
                "semantic": sem_score,
                "metadata": meta_score,
                "evidence": ev_score,
            }

            overall = sum(
                self.weights[k] * component_scores[k] for k in self.weights
            )

            # Find matching keywords
            doc_token_set = set(docs_tokens[idx])
            matching = [qt for qt in query_tokens if qt in doc_token_set]

            # Find relevant objectives
            objectives = inv.get("objectives", [])
            relevant_objs: list[str] = []
            if objectives:
                query_lower = query.lower()
                for obj in objectives:
                    if any(kw in obj.lower() for kw in query_tokens):
                        relevant_objs.append(obj)

            explanation = self._build_explanation(
                query, inv, component_scores, matching
            )

            results.append(
                RankedResult(
                    investigation_id=inv_id,
                    title=title_val,
                    overall_score=round(min(max(overall, 0.0), 1.0), 4),
                    component_scores={
                        k: round(v, 4) for k, v in component_scores.items()
                    },
                    explanation=explanation,
                    matching_keywords=matching,
                    relevant_objectives=relevant_objs[:5],
                )
            )

        results.sort(key=lambda r: r.overall_score, reverse=True)
        return results[:top_k]

    def explain_ranking(
        self, query: str, investigation: dict[str, Any]
    ) -> RankingExplanation:
        """Produce a detailed explanation for a single investigation's rank.

        Args:
            query: The search query.
            investigation: Investigation dict.

        Returns:
            :class:`RankingExplanation` with per-component scores,
            top contributing factors, and matching text snippets.
        """
        query_tokens = _tokenize(query)
        doc_tokens = _tokenize(investigation.get("text", ""))
        avg_dl = float(len(doc_tokens)) or 1.0
        n_docs = 1

        idf = {qt: 1.0 for qt in query_tokens}

        kw = self.keyword_score(
            query, investigation.get("text", ""), query_tokens, doc_tokens, avg_dl, idf
        )
        sem = self.semantic_score(query, investigation.get("text", ""))
        meta = self.metadata_score(query, investigation)
        ev = self.evidence_score(investigation)

        scores = {
            "keyword": round(kw, 4),
            "semantic": round(sem, 4),
            "metadata": round(meta, 4),
            "evidence": round(ev, 4),
        }

        # Identify top factors
        factors: list[str] = []
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for comp, val in sorted_scores:
            if val > 0:
                factors.append(f"{comp}: {val:.3f}")

        # Extract matching snippets
        snippets = self._extract_snippets(query_tokens, investigation.get("text", ""))

        return RankingExplanation(
            investigation_id=investigation.get("investigation_id", "unknown"),
            query=query,
            scores=scores,
            top_factors=factors,
            matching_text_snippets=snippets,
        )

    # ------------------------------------------------------------------
    # Component scorers
    # ------------------------------------------------------------------

    def keyword_score(
        self,
        query: str,
        text: str,
        query_tokens: list[str] | None = None,
        doc_tokens: list[str] | None = None,
        avg_dl: float | None = None,
        idf: dict[str, float] | None = None,
    ) -> float:
        """BM25-inspired keyword matching score.

        Normalised to ``[0, 1]`` via sigmoid squashing.

        Args:
            query: Query string.
            text: Document text.
            query_tokens: Pre-tokenized query (optional optimisation).
            doc_tokens: Pre-tokenized document (optional optimisation).
            avg_dl: Average document length (for BM25).
            idf: Pre-computed IDF dict.

        Returns:
            Score in ``[0, 1]``.
        """
        if not text:
            return 0.0

        qt = query_tokens if query_tokens is not None else _tokenize(query)
        dt = doc_tokens if doc_tokens is not None else _tokenize(text)

        if not qt or not dt:
            return 0.0

        adl = avg_dl if avg_dl is not None else float(len(dt))
        if idf is None:
            idf = {t: 1.0 for t in qt}

        raw = _bm25_score(qt, dt, adl, idf)

        # Sigmoid normalisation: score = 2 * sigmoid(raw) - 1 mapped to [0,1]
        normalised = 1.0 / (1.0 + math.exp(-raw / max(len(qt), 1)))
        return float(normalised)

    def semantic_score(self, query: str, text: str) -> float:
        """Cosine similarity via embeddings.

        Returns:
            Score in ``[0, 1]`` (clamped from raw cosine).
        """
        if not text:
            return 0.0
        try:
            sim = self.embeddings.similarity(query, text)
            return float(max(0.0, min(1.0, sim)))
        except (RuntimeError, ValueError):
            return 0.0

    def metadata_score(self, query: str, investigation: dict[str, Any]) -> float:
        """Score based on category/platform/status relevance.

        Heuristic matching of query terms against structured metadata.

        Returns:
            Score in ``[0, 1]``.
        """
        query_lower = query.lower()
        score = 0.0
        match_count = 0
        max_checks = 0

        # Category matching
        categories = investigation.get("categories", [])
        if categories:
            max_checks += 1
            for cat in categories:
                # Convert snake_case to words
                cat_words = cat.replace("_", " ")
                if any(w in query_lower for w in cat_words.split()):
                    match_count += 1
                    break

        # Platform matching
        platform = investigation.get("platform", "")
        if platform:
            max_checks += 1
            for plat_key, keywords in self.PLATFORM_KEYWORDS.items():
                if plat_key.lower() in platform.lower():
                    if any(kw in query_lower for kw in keywords):
                        match_count += 1
                        break

        # Status matching
        status = investigation.get("status", "")
        if status:
            max_checks += 1
            for stat_key, keywords in self.STATUS_KEYWORDS.items():
                if stat_key.lower() in status.lower():
                    if any(kw in query_lower for kw in keywords):
                        match_count += 1
                        break

        if max_checks == 0:
            return 0.5  # neutral when no metadata available

        return match_count / max_checks

    def evidence_score(self, investigation: dict[str, Any]) -> float:
        """Score based on data quality and completeness.

        Considers:
            - ``data_completeness`` (0-1 float)
            - ``version_count`` (int, more versions → more mature)
            - Presence of ``results``, ``publications`` fields

        Returns:
            Score in ``[0, 1]``.
        """
        sub_scores: list[float] = []

        # Data completeness
        completeness = investigation.get("data_completeness")
        if completeness is not None:
            sub_scores.append(float(completeness))

        # Version count (log-scaled, cap at 10)
        version_count = investigation.get("version_count")
        if version_count is not None:
            vc = max(int(version_count), 0)
            sub_scores.append(min(math.log2(vc + 1) / math.log2(11), 1.0))

        # Presence of results
        if investigation.get("results"):
            sub_scores.append(1.0)
        elif "results" in investigation:
            sub_scores.append(0.3)

        # Presence of publications
        pubs = investigation.get("publications", [])
        if pubs:
            sub_scores.append(min(len(pubs) / 5.0, 1.0))

        # Text length as a proxy for documentation quality
        text = investigation.get("text", "")
        if text:
            word_count = len(text.split())
            sub_scores.append(min(word_count / 500.0, 1.0))

        if not sub_scores:
            return 0.5  # neutral default
        return float(np.mean(sub_scores))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_explanation(
        self,
        query: str,
        investigation: dict[str, Any],
        scores: dict[str, float],
        matching_keywords: list[str],
    ) -> str:
        """Build a human-readable one-line explanation."""
        parts: list[str] = []

        best_component = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_val = scores[best_component]

        if matching_keywords:
            kw_str = ", ".join(matching_keywords[:5])
            parts.append(f"keyword matches: [{kw_str}]")

        if scores["semantic"] > 0.5:
            parts.append("strong semantic similarity")
        elif scores["semantic"] > 0.3:
            parts.append("moderate semantic similarity")

        if scores["metadata"] > 0.5:
            parts.append("metadata aligns with query")

        if scores["evidence"] > 0.7:
            parts.append("high data quality")

        if not parts:
            parts.append(f"best signal from {best_component} ({best_val:.2f})")

        return "; ".join(parts)

    @staticmethod
    def _extract_snippets(
        query_tokens: list[str], text: str, max_snippets: int = 3
    ) -> list[str]:
        """Extract short text snippets around query-term matches."""
        if not text or not query_tokens:
            return []

        sentences = re.split(r"[.!?]+", text)
        snippets: list[str] = []
        query_set = set(query_tokens)

        for sent in sentences:
            sent_stripped = sent.strip()
            if not sent_stripped:
                continue
            sent_tokens = set(_tokenize(sent_stripped))
            overlap = query_set & sent_tokens
            if overlap:
                # Truncate long sentences
                snippet = sent_stripped[:200] + ("..." if len(sent_stripped) > 200 else "")
                snippets.append(snippet)
                if len(snippets) >= max_snippets:
                    break

        return snippets
