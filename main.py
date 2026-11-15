#!/usr/bin/env python3
"""
Flame in Freefall — AI-Powered Space Combustion Research Intelligence CLI.

Sub-commands:
    python main.py ingest      — Run NASA PSI ingestion and provenance versioning.
    python main.py classify    — Train and run research classification engine.
    python main.py search "q"  — Perform faceted hybrid search over investigations.
    python main.py evaluate    — Evaluate classification (Macro F1) & ranking (NDCG@5).
    python main.py dashboard   — Launch the interactive research intelligence dashboard.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import ProjectConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("flame-in-freefall")


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Run data ingestion and version tracking."""
    cfg = ProjectConfig()
    cfg.ensure_directories()
    from src.ingestion.psi_catalog import PSICatalogIngester

    logger.info("Starting NASA PSI ingestion pipeline...")
    ingester = PSICatalogIngester(cfg)
    investigations = ingester.ingest_from_seed()
    completeness = ingester.validate_completeness(investigations)

    print("\n" + "=" * 65)
    print("FLAME IN FREEFALL — DATA INGESTION REPORT")
    print("=" * 65)
    print(f"Total investigations indexed : {len(investigations)}")
    print(f"Mean completeness score       : {completeness['mean_completeness'] * 100:.1f}%")
    print(f"Active versions verified      : {sum(1 for i in investigations if i.active_version is not None)}")
    print(f"DOI citations tracked         : {sum(1 for i in investigations if i.get_active_version() and i.get_active_version().doi)}")
    print("-" * 65)
    for inv in investigations:
        active_ver = inv.get_active_version()
        doi = str(active_ver.doi) if (active_ver and active_ver.doi) else "N/A"
        cats = ", ".join(c.value for c in inv.categories)
        print(f"• [{inv.id:<12}] v{inv.active_version} | DOI: {doi:<18} | {inv.title[:35]}")
    print("=" * 65 + "\n")
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    """Train classifier and assign predicted categories."""
    cfg = ProjectConfig()
    cfg.ensure_directories()
    from src.catalog.repository import ResearchCatalog
    from src.ai.classifier import ResearchClassifier

    catalog = ResearchCatalog()
    investigations = catalog.list_all()
    if not investigations:
        logger.warning("Catalogue empty. Running ingestion first...")
        _cmd_ingest(args)
        investigations = catalog.list_all()

    labels_file = cfg.processed_dir / "training_labels.json"
    if not labels_file.exists():
        logger.error(f"Training labels not found at {labels_file}")
        return 1

    with open(labels_file, "r", encoding="utf-8") as f:
        training_data = json.load(f)

    logger.info("Training ResearchClassifier on %d curated records...", len(training_data))
    clf = ResearchClassifier()
    metrics = clf.train(training_data)

    model_path = cfg.models_dir / "classifier.pkl"
    clf.save(model_path)
    logger.info("Saved trained classifier model to %s", model_path)

    print("\n" + "=" * 65)
    print("RESEARCH CLASSIFICATION ENGINE — TRAINING RESULTS")
    print("=" * 65)
    print(f"Macro F1 Score (Cross-Val)   : {metrics['macro_f1']:.3f}")
    print(f"Macro Precision              : {metrics['macro_precision']:.3f}")
    print(f"Macro Recall                 : {metrics['macro_recall']:.3f}")
    print("-" * 65)
    print("Classifying catalogue investigations...")

    for inv in investigations:
        preds = clf.predict(inv.to_search_text())
        conf_dict = {cat: round(score, 3) for cat, score in preds}
        pred_enums = []
        for cat, _ in preds:
            from src.catalog.schema import ResearchCategory
            pred_enums.append(ResearchCategory.from_string(cat))

        inv.predicted_categories = pred_enums
        inv.category_confidence = conf_dict
        catalog.update(inv)

    print(f"Successfully classified and updated {len(investigations)} catalogue records.")
    print("=" * 65 + "\n")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    """Run search engine over investigations."""
    from src.rag.search_engine import UnifiedSearchEngine, SearchFilters

    engine = UnifiedSearchEngine()
    filters = SearchFilters(
        categories=[args.category] if getattr(args, "category", None) else None,
        platforms=[args.platform] if getattr(args, "platform", None) else None,
    )

    res = engine.search(query=args.query, filters=filters, top_k=args.top_k)

    print("\n" + "=" * 65)
    print(f"SEARCH RESULTS for: \"{args.query}\" ({res.total_results} matches in {res.search_time_ms} ms)")
    print("=" * 65)
    if not res.results:
        print("No matching investigations found.")
    for i, r in enumerate(res.results, start=1):
        doi_str = f"DOI: {r.doi}" if r.doi else "No DOI"
        print(f"{i}. [{r.investigation_id}] {r.title} (Score: {r.score})")
        print(f"   Platform: {r.platform.upper()} | {doi_str}")
        print(f"   Categories: {', '.join(r.categories)}")
        if r.highlights:
            print(f"   Signals: {', '.join(r.highlights[:2])}")
        print()
    print("=" * 65 + "\n")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Run rigorous scientific evaluation across all AI subsystems."""
    cfg = ProjectConfig()
    from src.ai.classifier import ResearchClassifier
    from src.ai.embeddings import EmbeddingsManager
    from src.ai.ranker import ResearchRanker
    from src.ai.evaluation import AIEvaluator
    from src.catalog.repository import ResearchCatalog
    from src.rag.chunker import DocumentChunker
    from src.rag.retriever import InvestigationRetriever
    from src.rag.assistant import ResearchAssistant

    labels_file = cfg.processed_dir / "training_labels.json"
    queries_file = cfg.processed_dir / "test_queries.json"

    with open(labels_file, "r", encoding="utf-8") as f:
        training_data = json.load(f)
    with open(queries_file, "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    catalog = ResearchCatalog()
    catalog_invs = [inv.model_dump() for inv in catalog.list_all()]

    clf = ResearchClassifier()
    cv_metrics = clf.train(training_data)

    em = EmbeddingsManager()
    em.fit([inv["text"] for inv in training_data])
    ranker = ResearchRanker(embeddings=em)

    chunker = DocumentChunker()
    chunks = chunker.chunk_all(catalog_invs)
    retriever = InvestigationRetriever()
    retriever.build_index(chunks)
    assistant = ResearchAssistant(retriever=retriever, catalog=catalog)

    evaluator = AIEvaluator()
    cls_report = evaluator.evaluate_classifier(clf, training_data)
    rank_report = evaluator.evaluate_ranker(ranker, test_queries, catalog_invs)
    rag_report = evaluator.evaluate_rag(assistant, test_queries, use_llm=False)

    print("\n" + "=" * 70)
    print("FLAME IN FREEFALL — SCIENTIFIC EVALUATION BENCHMARK REPORT")
    print("=" * 70)
    print(f"Evaluation Timestamp          : {cls_report.timestamp}")
    print(f"Total Benchmark Queries       : {rank_report.n_queries}")
    print(f"Total Evaluated Investigations: {cls_report.n_samples}")
    print("-" * 70)
    print("1. MODEL 1: RESEARCH CLASSIFICATION ENGINE (Multi-Label)")
    print(f"   • Cross-Val Macro F1 (Stratified) : {cv_metrics.get('macro_f1', 0.0):.4f} (Indicative, 16 flight records)")
    print(f"   • In-Sample Macro F1              : {cls_report.macro_f1:.4f}")
    print(f"   • Cross-Val Macro Precision       : {cv_metrics.get('macro_precision', 0.0):.4f}")
    print(f"   • Cross-Val Macro Recall          : {cv_metrics.get('macro_recall', 0.0):.4f}")
    print("-" * 70)
    print("2. MODEL 2: RELEVANCE RANKING ENGINE (Hybrid BM25 + Semantic)")
    print(f"   • Mean Precision@5          : {rank_report.mean_precision_at_k:.4f}")
    print(f"   • Mean NDCG@5 (Rank Quality): {rank_report.mean_ndcg_at_k:.4f}")
    print("-" * 70)
    active_dois = sum(1 for inv in catalog.list_all() if inv.get_active_version() and inv.get_active_version().doi)
    total_invs = len(catalog_invs)
    doi_pct = (active_dois / total_invs * 100) if total_invs else 0.0

    print("3. RESEARCH RETRIEVAL & CITATION TRACEABILITY")
    print(f"   • Provenance Traceability   : {doi_pct:.1f}% of indexed records have active DOIs ({active_dois}/{total_invs})")
    print("-" * 70)
    print("4. CONVERSATIONAL RAG & EVIDENCE GROUNDING")
    print(f"   • Benchmark Status          : {rag_report.benchmark_status}")
    print(f"   • Citation Validity Rate    : {rag_report.citation_validity_rate:.4f}")
    print(f"   • Evidence Relevance Score  : {rag_report.evidence_relevance_score:.4f}")
    print(f"   • Answer Groundedness Rate  : {rag_report.groundedness_rate:.4f}")
    print(f"   • Hallucination Rate        : {rag_report.hallucination_rate:.4f}")
    print(f"   • Abstention Accuracy       : {rag_report.abstention_accuracy:.4f}")
    print("=" * 70 + "\n")
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    """Launch the interactive dashboard."""
    port = getattr(args, "port", 8501)
    
    # Check if streamlit is available
    has_streamlit = False
    try:
        import streamlit
        has_streamlit = True
    except ImportError:
        has_streamlit = False

    if has_streamlit:
        dashboard_app = _PROJECT_ROOT / "app.py"
        logger.info("Launching Streamlit dashboard on port %d...", port)
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_app), "--server.port", str(port)])
    else:
        logger.info("Streamlit not detected in current environment. Launching built-in interactive web UI on port %d...", port)
        from dashboard.standalone_server import run_standalone_dashboard
        run_standalone_dashboard(port=port)
    return 0



def _cmd_llm_check(args: argparse.Namespace) -> int:
    """Check Ollama LLM server connectivity and model availability."""
    cfg = ProjectConfig()
    from src.rag.ollama_client import OllamaClient

    client = OllamaClient(cfg.llm)
    health = client.check_health()

    print()
    print("=" * 68)
    print("FLAME IN FREEFALL — OLLAMA LLM RUNTIME DIAGNOSTIC")
    print("=" * 68)
    print(f"Provider            : {cfg.llm.provider.upper()}")
    print(f"Base URL            : {cfg.llm.base_url}")
    print(f"Configured Model    : {cfg.llm.model}")
    print(f"Timeout             : {cfg.llm.timeout_seconds}s")
    print(f"Temperature         : {cfg.llm.temperature}")
    auth_status = "Configured (hidden)" if cfg.llm.auth_header else "None (Standard/Local)"
    print(f"Auth Header         : {auth_status}")
    print("-" * 68)

    status = health.get("status")
    if status == "online":
        print("Server Connectivity : ONLINE (Connected successfully)")
        models = health.get("available_models", [])
        print(f"Available Models ({len(models)}): {', '.join(models) or 'None installed yet'}")
        if health.get("model_available"):
            print(f"Target Model Status : READY ('{cfg.llm.model}' is available)")
            print("\nTesting quick inference probe...")
            try:
                probe_resp = client.chat(
                    messages=[{"role": "user", "content": "Say Assistant Online in 3 words."}],
                    temperature=0.0,
                )
                print(f"Probe Response      : '{probe_resp.get('content')}'")
                print("Result              : LLM Subsystem 100% Operational")
            except Exception as e:
                print(f"Probe Failed        : [ERROR] {e}")
        else:
            print(f"Target Model Status : MISSING ('{cfg.llm.model}' is not pulled on the server)")
            print(f"Action Required     : Run `ollama pull {cfg.llm.model}` on your Ollama host.")
    else:
        print(f"Server Connectivity : OFFLINE ({health.get('reason', 'Connection failed')})")
        print("\nTroubleshooting Tips:")
        print("  1. If running locally:")
        print(f"     Ensure Ollama is running (`ollama serve`) and pull your model: `ollama pull {cfg.llm.model}`")
        print("  2. If running Ollama in Google Colab:")
        print("     Configuration A (Dashboard inside Colab): export OLLAMA_BASE_URL=http://127.0.0.1:11434")
        print("     Configuration B (Remote tunnel to Colab): export OLLAMA_BASE_URL=https://<your-tunnel-url>")
        print("     If tunnel requires authentication: export OLLAMA_AUTH_HEADER=\"Bearer <token>\"")
        print("  3. Note: The platform automatically uses verified deterministic fallback mode when offline.")

    print("=" * 68)
    print()
    return 0 if (status == "online" and health.get("model_available")) else 1



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flame-in-freefall",
        description="Flame in Freefall — AI-Powered Space Combustion Research Intelligence",
    )
    sub = parser.add_subparsers(dest="command", help="Available sub-commands")

    # Ingest
    sub.add_parser("ingest", help="Ingest NASA PSI records and build catalogue")

    # Classify
    sub.add_parser("classify", help="Train and execute AI classification model")

    # Search
    s_parser = sub.add_parser("search", help="Perform faceted search over investigations")
    s_parser.add_argument("query", type=str, help="Search query string")
    s_parser.add_argument("--category", type=str, default=None, help="Filter by category")
    s_parser.add_argument("--platform", type=str, default=None, help="Filter by platform")
    s_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")

    # Evaluate
    sub.add_parser("evaluate", help="Run benchmark evaluation (Macro F1, NDCG@5, Precision@5)")

    # Dashboard
    d_parser = sub.add_parser("dashboard", help="Launch interactive research intelligence dashboard")
    d_parser.add_argument("--port", type=int, default=8501, help="Port to bind dashboard server")

    # LLM Check
    sub.add_parser("llm-check", help="Test Ollama connection, model availability, and runtime diagnostic")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "ingest": _cmd_ingest,
        "classify": _cmd_classify,
        "search": _cmd_search,
        "evaluate": _cmd_evaluate,
        "dashboard": _cmd_dashboard,
        "llm-check": _cmd_llm_check,
    }

    handler = commands.get(args.command)
    if not handler:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
