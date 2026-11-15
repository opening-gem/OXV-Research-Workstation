"""
AI Evaluation Framework for Flame in Freefall.

Provides evaluation tools for the classifier, ranker, and search
subsystems, including standard IR metrics such as Precision@k,
NDCG@k, macro-F1, and per-category precision/recall/F1.
"""

from __future__ import annotations

import math
import logging
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
from src.catalog.compat_pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ======================================================================
# Report models
# ======================================================================


class CategoryMetrics(BaseModel):
    """Per-category classification metrics."""

    category: str
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0


class ClassificationReport(BaseModel):
    """Full classification evaluation report."""

    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    per_category: list[CategoryMetrics] = Field(default_factory=list)
    n_samples: int = 0
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class QueryEvaluation(BaseModel):
    """Evaluation of a single ranking query."""

    query: str
    precision_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    k: int = 5
    retrieved_ids: list[str] = Field(default_factory=list)
    relevant_ids: list[str] = Field(default_factory=list)


class RankingReport(BaseModel):
    """Full ranking evaluation report."""

    mean_precision_at_k: float = 0.0
    mean_ndcg_at_k: float = 0.0
    k: int = 5
    n_queries: int = 0
    per_query: list[QueryEvaluation] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class SearchReport(BaseModel):
    """Search quality evaluation report."""

    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_f1: float = 0.0
    n_queries: int = 0
    per_query: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class RAGQueryEvaluation(BaseModel):
    """Evaluation of a single RAG assistant query."""

    query: str
    retrieved_ids: list[str] = Field(default_factory=list)
    cited_ids: list[str] = Field(default_factory=list)
    validated_citations: list[str] = Field(default_factory=list)
    rejected_citations: list[str] = Field(default_factory=list)
    citation_validity_rate: float = 1.0
    evidence_relevance_score: float = 0.0
    grounded: bool = True
    abstention_appropriate: bool = True
    llm_generated: bool = False


class RAGEvaluationReport(BaseModel):
    """Evaluation report for RAG generation, citation validity, and groundedness.

    Distinguishes retrieval metrics from generation/grounding quality.
    All small-sample benchmarks are explicitly labeled as preliminary.
    """

    n_queries: int = 0
    citation_validity_rate: float = 1.0
    evidence_relevance_score: float = 0.0
    groundedness_rate: float = 1.0
    hallucination_rate: float = 0.0
    abstention_accuracy: float = 1.0
    per_query: list[RAGQueryEvaluation] = Field(default_factory=list)
    benchmark_status: str = "PRELIMINARY (16 flight records, 10 queries)"
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ======================================================================
# Metric helpers
# ======================================================================


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute Precision@k.

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant: Set of relevant document IDs.
        k: Cutoff.

    Returns:
        Precision@k in ``[0, 1]``.
    """
    if k <= 0:
        return 0.0
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / k


def dcg_at_k(
    retrieved: list[str],
    relevance_grades: dict[str, int],
    k: int,
) -> float:
    """Compute Discounted Cumulative Gain at k.

    Uses the standard formula:
        DCG@k = sum_{i=1}^{k} (2^{rel_i} - 1) / log2(i + 1)

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevance_grades: Mapping of document ID → relevance grade
            (higher is more relevant).
        k: Cutoff.

    Returns:
        DCG@k value.
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        rel = relevance_grades.get(doc_id, 0)
        dcg += (2.0 ** rel - 1.0) / math.log2(i + 2)  # i+2 because i is 0-indexed
    return dcg


def ndcg_at_k(
    retrieved: list[str],
    relevance_grades: dict[str, int],
    k: int,
) -> float:
    """Compute Normalised Discounted Cumulative Gain at k.

    NDCG@k = DCG@k / IDCG@k, where IDCG@k is the DCG of the ideal
    ranking (documents sorted by descending relevance grade).

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevance_grades: Mapping of document ID → relevance grade.
        k: Cutoff.

    Returns:
        NDCG@k in ``[0, 1]``. Returns 0.0 when IDCG is 0.
    """
    actual_dcg = dcg_at_k(retrieved, relevance_grades, k)

    # Ideal ranking: all graded docs sorted by descending relevance
    ideal_order = sorted(relevance_grades.keys(), key=lambda d: relevance_grades[d], reverse=True)
    ideal_dcg = dcg_at_k(ideal_order, relevance_grades, k)

    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


# ======================================================================
# Main evaluator
# ======================================================================


class AIEvaluator:
    """Evaluates the AI subsystems of Flame in Freefall.

    Provides methods for evaluating the classifier, ranker, and search
    components, plus a combined human-readable report generator.

    Example:
        >>> evaluator = AIEvaluator()
        >>> cls_report = evaluator.evaluate_classifier(classifier, test_data)
        >>> rank_report = evaluator.evaluate_ranker(ranker, test_queries, investigations)
        >>> print(evaluator.generate_report())
    """

    def __init__(self) -> None:
        self._classification_report: ClassificationReport | None = None
        self._ranking_report: RankingReport | None = None
        self._search_report: SearchReport | None = None
        self._rag_report: RAGEvaluationReport | None = None

    # ------------------------------------------------------------------
    # Classifier evaluation
    # ------------------------------------------------------------------

    def evaluate_classifier(
        self,
        classifier: Any,
        test_data: list[dict[str, Any]],
    ) -> ClassificationReport:
        """Evaluate a trained :class:`ResearchClassifier`.

        Args:
            classifier: A trained ``ResearchClassifier`` instance.
            test_data: List of dicts with ``text`` and ``categories``.

        Returns:
            :class:`ClassificationReport` with macro and per-category
            metrics.
        """
        if not test_data:
            raise ValueError("test_data must not be empty")

        # Use the classifier's own evaluate method
        raw = classifier.evaluate(test_data)

        per_category: list[CategoryMetrics] = []
        for cat, metrics in raw.get("per_category", {}).items():
            per_category.append(
                CategoryMetrics(
                    category=cat,
                    precision=metrics.get("precision", 0.0),
                    recall=metrics.get("recall", 0.0),
                    f1=metrics.get("f1", 0.0),
                    support=metrics.get("support", 0),
                )
            )

        report = ClassificationReport(
            macro_precision=raw.get("macro_precision", 0.0),
            macro_recall=raw.get("macro_recall", 0.0),
            macro_f1=raw.get("macro_f1", 0.0),
            per_category=per_category,
            n_samples=raw.get("n_test_samples", len(test_data)),
        )
        self._classification_report = report
        return report

    # ------------------------------------------------------------------
    # Ranker evaluation
    # ------------------------------------------------------------------

    def evaluate_ranker(
        self,
        ranker: Any,
        queries_with_relevance: list[dict[str, Any]],
        investigations: list[dict[str, Any]],
        k: int = 5,
    ) -> RankingReport:
        """Evaluate the :class:`ResearchRanker`.

        Args:
            ranker: A ``ResearchRanker`` instance.
            queries_with_relevance: List of dicts, each containing:
                - ``query`` (str)
                - ``relevant_investigations`` (list[str])
                - ``relevance_grades`` (dict[str, int])
            investigations: Full list of investigation dicts to rank.
            k: Cutoff for Precision@k and NDCG@k.

        Returns:
            :class:`RankingReport` with mean P@k, NDCG@k, and
            per-query breakdowns.
        """
        if not queries_with_relevance:
            raise ValueError("queries_with_relevance must not be empty")

        per_query: list[QueryEvaluation] = []
        all_p_at_k: list[float] = []
        all_ndcg: list[float] = []

        for qr in queries_with_relevance:
            query = qr["query"]
            relevant_ids = set(qr.get("relevant_investigations", []))
            rel_grades = qr.get("relevance_grades", {})

            # If no grades provided, assign grade 1 for all relevant
            if not rel_grades and relevant_ids:
                rel_grades = {rid: 1 for rid in relevant_ids}

            # Rank
            results = ranker.rank(query, investigations, top_k=k)
            retrieved = [r.investigation_id for r in results]

            p_k = precision_at_k(retrieved, relevant_ids, k)
            n_k = ndcg_at_k(retrieved, rel_grades, k)

            all_p_at_k.append(p_k)
            all_ndcg.append(n_k)

            per_query.append(
                QueryEvaluation(
                    query=query,
                    precision_at_k=round(p_k, 4),
                    ndcg_at_k=round(n_k, 4),
                    k=k,
                    retrieved_ids=retrieved,
                    relevant_ids=list(relevant_ids),
                )
            )

        report = RankingReport(
            mean_precision_at_k=round(float(np.mean(all_p_at_k)), 4) if all_p_at_k else 0.0,
            mean_ndcg_at_k=round(float(np.mean(all_ndcg)), 4) if all_ndcg else 0.0,
            k=k,
            n_queries=len(queries_with_relevance),
            per_query=per_query,
        )
        self._ranking_report = report
        return report

    # ------------------------------------------------------------------
    # Search evaluation
    # ------------------------------------------------------------------

    def evaluate_search(
        self,
        search_fn: Callable[[str], list[str]],
        queries_with_expected: list[dict[str, Any]],
    ) -> SearchReport:
        """Evaluate a search function.

        Args:
            search_fn: Callable that takes a query string and returns a
                list of investigation IDs.
            queries_with_expected: List of dicts with:
                - ``query`` (str)
                - ``expected`` (list[str]): expected investigation IDs.

        Returns:
            :class:`SearchReport` with mean precision, recall, F1.
        """
        if not queries_with_expected:
            raise ValueError("queries_with_expected must not be empty")

        per_query: list[dict[str, Any]] = []
        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []

        for qe in queries_with_expected:
            query = qe["query"]
            expected = set(qe.get("expected", []))
            retrieved = search_fn(query)
            retrieved_set = set(retrieved)

            if not retrieved_set and not expected:
                p, r, f1 = 1.0, 1.0, 1.0
            elif not retrieved_set:
                p, r, f1 = 0.0, 0.0, 0.0
            elif not expected:
                p, r, f1 = 0.0, 1.0, 0.0
            else:
                hits = retrieved_set & expected
                p = len(hits) / len(retrieved_set)
                r = len(hits) / len(expected)
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)

            per_query.append(
                {
                    "query": query,
                    "precision": round(p, 4),
                    "recall": round(r, 4),
                    "f1": round(f1, 4),
                    "retrieved": retrieved,
                    "expected": list(expected),
                }
            )

        report = SearchReport(
            mean_precision=round(float(np.mean(precisions)), 4),
            mean_recall=round(float(np.mean(recalls)), 4),
            mean_f1=round(float(np.mean(f1s)), 4),
            n_queries=len(queries_with_expected),
            per_query=per_query,
        )
        self._search_report = report
        return report

    # ------------------------------------------------------------------
    # RAG & Evidence Grounding evaluation
    # ------------------------------------------------------------------

    def evaluate_rag(
        self,
        assistant: Any,
        queries_with_relevance: list[dict[str, Any]],
        use_llm: bool = False,
    ) -> RAGEvaluationReport:
        """Evaluate RAG generation, citation validity, and hallucination defense.

        Distinguishes IR/retrieval evaluation from LLM generation metrics.
        Evaluates citation validity, evidence relevance, groundedness, and abstention.

        Parameters
        ----------
        assistant : ResearchAssistant
            The research assistant instance to evaluate.
        queries_with_relevance : list[dict[str, Any]]
            Benchmark queries with optional 'relevant_investigations' and
            optional 'is_out_of_domain'.
        use_llm : bool, default=False
            Whether to query LLM or evaluate retrieval & citation pipeline.
        """
        if not queries_with_relevance:
            raise ValueError("queries_with_relevance must not be empty")

        per_query: list[RAGQueryEvaluation] = []
        c_validity_scores: list[float] = []
        ev_rel_scores: list[float] = []
        grounded_scores: list[float] = []
        abstention_scores: list[float] = []

        for q_item in queries_with_relevance:
            query = q_item["query"]
            relevant_set = set(q_item.get("relevant_investigations", []))
            is_ood = q_item.get("is_out_of_domain", False)

            resp = assistant.answer(query, use_llm=use_llm)

            retrieved_ids = [r.chunk.investigation_id for r in resp.retrieved_chunks]
            validated = list(resp.validated_citations)
            rejected = list(resp.rejected_citations)
            cited_all = list(dict.fromkeys(validated + rejected))

            # Citation validity: fraction of cited IDs that were verified in retrieved chunks
            if cited_all:
                c_val = len(validated) / len(cited_all)
            else:
                c_val = 1.0

            # Evidence relevance: fraction of retrieved chunks matching ground truth
            unique_retrieved = list(dict.fromkeys(retrieved_ids))
            if relevant_set and unique_retrieved:
                hits = sum(1 for rid in unique_retrieved if rid in relevant_set)
                ev_rel = hits / len(unique_retrieved)
            elif is_ood:
                ev_rel = 1.0 if not unique_retrieved or resp.confidence == "insufficient_evidence" else 0.0
            else:
                ev_rel = 0.5

            # Groundedness: 0 rejected / ungrounded citations
            is_grounded = len(rejected) == 0

            # Abstention appropriateness
            if is_ood:
                abstention_ok = (resp.confidence == "insufficient_evidence")
            else:
                abstention_ok = (resp.confidence != "insufficient_evidence")

            c_validity_scores.append(c_val)
            ev_rel_scores.append(ev_rel)
            grounded_scores.append(1.0 if is_grounded else 0.0)
            abstention_scores.append(1.0 if abstention_ok else 0.0)

            per_query.append(
                RAGQueryEvaluation(
                    query=query,
                    retrieved_ids=unique_retrieved,
                    cited_ids=cited_all,
                    validated_citations=validated,
                    rejected_citations=rejected,
                    citation_validity_rate=round(c_val, 4),
                    evidence_relevance_score=round(ev_rel, 4),
                    grounded=is_grounded,
                    abstention_appropriate=abstention_ok,
                    llm_generated=resp.llm_generated,
                )
            )

        mean_c_val = float(np.mean(c_validity_scores)) if c_validity_scores else 1.0
        mean_ev_rel = float(np.mean(ev_rel_scores)) if ev_rel_scores else 0.0
        mean_grounded = float(np.mean(grounded_scores)) if grounded_scores else 1.0
        halluc_rate = 1.0 - mean_c_val
        mean_abstention = float(np.mean(abstention_scores)) if abstention_scores else 1.0

        report = RAGEvaluationReport(
            n_queries=len(queries_with_relevance),
            citation_validity_rate=round(mean_c_val, 4),
            evidence_relevance_score=round(mean_ev_rel, 4),
            groundedness_rate=round(mean_grounded, 4),
            hallucination_rate=round(halluc_rate, 4),
            abstention_accuracy=round(mean_abstention, 4),
            per_query=per_query,
            benchmark_status="PRELIMINARY (16 flight records, 10 retrieval queries)",
        )
        self._rag_report = report
        return report

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self) -> str:
        """Generate a human-readable evaluation report.

        Combines results from classifier, ranker, search, and RAG
        evaluations that have been run.

        Returns:
            Formatted multi-line report string.
        """
        lines: list[str] = [
            "=" * 60,
            "  Flame in Freefall — AI Evaluation Report",
            f"  Generated: {datetime.now(timezone.utc).isoformat()}",
            "=" * 60,
            "",
        ]

        # Classification
        if self._classification_report:
            cr = self._classification_report
            lines.append("─── Classification ─────────────────────────────")
            lines.append(f"  Samples: {cr.n_samples}")
            lines.append(f"  Macro Precision: {cr.macro_precision:.4f}")
            lines.append(f"  Macro Recall:    {cr.macro_recall:.4f}")
            lines.append(f"  Macro F1:        {cr.macro_f1:.4f}")
            lines.append("")
            lines.append(f"  {'Category':<30} {'P':>6} {'R':>6} {'F1':>6} {'Sup':>5}")
            lines.append("  " + "-" * 55)
            for cm in cr.per_category:
                lines.append(
                    f"  {cm.category:<30} {cm.precision:>6.3f} "
                    f"{cm.recall:>6.3f} {cm.f1:>6.3f} {cm.support:>5d}"
                )
            lines.append("")
        else:
            lines.append("─── Classification ─────────────────────────────")
            lines.append("  (not evaluated)")
            lines.append("")

        # Ranking
        if self._ranking_report:
            rr = self._ranking_report
            lines.append("─── Ranking ────────────────────────────────────")
            lines.append(f"  Queries:         {rr.n_queries}")
            lines.append(f"  k:               {rr.k}")
            lines.append(f"  Mean P@{rr.k}:       {rr.mean_precision_at_k:.4f}")
            lines.append(f"  Mean NDCG@{rr.k}:    {rr.mean_ndcg_at_k:.4f}")
            lines.append("")
            for qe in rr.per_query:
                lines.append(f"  Q: \"{qe.query}\"")
                lines.append(
                    f"    P@{qe.k}={qe.precision_at_k:.3f}  "
                    f"NDCG@{qe.k}={qe.ndcg_at_k:.3f}  "
                    f"Retrieved: {qe.retrieved_ids[:5]}"
                )
            lines.append("")
        else:
            lines.append("─── Ranking ────────────────────────────────────")
            lines.append("  (not evaluated)")
            lines.append("")

        # Search
        if self._search_report:
            sr = self._search_report
            lines.append("─── Search ─────────────────────────────────────")
            lines.append(f"  Queries:         {sr.n_queries}")
            lines.append(f"  Mean Precision:  {sr.mean_precision:.4f}")
            lines.append(f"  Mean Recall:     {sr.mean_recall:.4f}")
            lines.append(f"  Mean F1:         {sr.mean_f1:.4f}")
            lines.append("")
        else:
            lines.append("─── Search ─────────────────────────────────────")
            lines.append("  (not evaluated)")
            lines.append("")

        # RAG Generation & Grounding
        if self._rag_report:
            rag = self._rag_report
            lines.append("─── RAG & Grounding ────────────────────────────")
            lines.append(f"  Status:                  {rag.benchmark_status}")
            lines.append(f"  Queries Evaluated:       {rag.n_queries}")
            lines.append(f"  Citation Validity Rate:  {rag.citation_validity_rate:.4f}")
            lines.append(f"  Evidence Relevance:      {rag.evidence_relevance_score:.4f}")
            lines.append(f"  Answer Groundedness:     {rag.groundedness_rate:.4f}")
            lines.append(f"  Hallucination Rate:      {rag.hallucination_rate:.4f}")
            lines.append(f"  Abstention Accuracy:     {rag.abstention_accuracy:.4f}")
            lines.append("  Note: Grounded verification matches cited IDs against retrieved context.")
            lines.append("")
        else:
            lines.append("─── RAG & Grounding ───────────────────────────")
            lines.append("  (not evaluated)")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)
