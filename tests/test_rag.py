"""Unit tests for RAG assistant, chunker, retriever, and unified search."""
import unittest

from src.rag.chunker import DocumentChunker
from src.rag.retriever import InvestigationRetriever
from src.rag.assistant import ResearchAssistant
from src.rag.search_engine import UnifiedSearchEngine, SearchFilters
from src.catalog.repository import ResearchCatalog


class TestRAGSubsystems(unittest.TestCase):
    def setUp(self):
        self.catalog = ResearchCatalog()
        self.invs = [inv.model_dump() for inv in self.catalog.list_all()]

    def test_chunking_and_retrieval(self):
        chunker = DocumentChunker()
        chunks = chunker.chunk_all(self.invs)
        self.assertTrue(len(chunks) > 0)

        retriever = InvestigationRetriever()
        retriever.build_index(chunks)
        results = retriever.retrieve("fire safety flame spread", top_k=3)
        self.assertTrue(len(results) > 0)
        self.assertTrue(results[0].score > 0)

    def test_assistant_answers_and_sources(self):
        chunker = DocumentChunker()
        chunks = chunker.chunk_all(self.invs)
        retriever = InvestigationRetriever()
        retriever.build_index(chunks)
        assistant = ResearchAssistant(retriever=retriever, catalog=self.catalog)

        # Comparison query
        resp = assistant.answer("What is the difference between ACME and BASS-II?")
        self.assertIsNotNone(resp.answer)
        self.assertTrue(len(resp.sources) > 0)
        self.assertIn(resp.confidence, ["high", "medium", "low", "insufficient_evidence"])

    def test_unified_search_engine(self):
        engine = UnifiedSearchEngine(catalog=self.catalog)
        res = engine.search(query="smoke", top_k=5)
        self.assertTrue(res.total_results > 0)
        self.assertEqual(res.results[0].investigation_id, "SAME")

        # Facets
        facets = engine.get_facets()
        self.assertIn("categories", facets)
        self.assertIn("platforms", facets)


if __name__ == "__main__":
    unittest.main()
