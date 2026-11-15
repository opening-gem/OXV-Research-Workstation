"""
Unit tests for Ollama LLM integration, RAG pipeline, and citation validation.
Uses mocked HTTP responses so that tests run reliably offline without a live Ollama server.
"""
import io
import json
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from src.catalog.repository import ResearchCatalog
from src.rag.assistant import ResearchAssistant
from src.rag.chunker import DocumentChunker
from src.rag.ollama_client import (
    LLMConfig,
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)
from src.rag.retriever import InvestigationRetriever


class MockHTTPResponse:
    """Mock for urllib.request.urlopen response."""

    def __init__(self, data: dict, status: int = 200):
        self.data = json.dumps(data).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TestOllamaIntegration(unittest.TestCase):
    def setUp(self):
        self.catalog = ResearchCatalog()
        self.invs = [inv.model_dump() for inv in self.catalog.list_all()]
        self.chunker = DocumentChunker()
        self.chunks = self.chunker.chunk_all(self.invs)
        self.retriever = InvestigationRetriever()
        self.retriever.build_index(self.chunks)

        self.config = LLMConfig(
            provider="ollama",
            base_url="http://127.0.0.1:11434",
            model="llama3",
            timeout_seconds=5,
            temperature=0.1,
        )
        self.client = OllamaClient(self.config)
        self.assistant = ResearchAssistant(
            retriever=self.retriever,
            catalog=self.catalog,
            llm_client=self.client,
        )

    @patch("urllib.request.urlopen")
    def test_successful_llm_response(self, mock_urlopen):
        """Test happy-path generation with verified citation extraction."""
        mock_payload = {
            "message": {
                "role": "assistant",
                "content": "BASS-II investigated solid fuel flame spread in microgravity aboard the ISS [BASS-II].",
            },
            "model": "llama3",
            "done": True,
        }
        mock_urlopen.return_value = MockHTTPResponse(mock_payload)

        resp = self.assistant.answer("What did BASS-II study?", use_llm=True)

        self.assertTrue(resp.llm_generated)
        self.assertEqual(resp.model_name, "llama3")
        self.assertIn("BASS-II", resp.validated_citations)
        self.assertEqual(len(resp.rejected_citations), 0)
        self.assertTrue(len(resp.sources) > 0)
        self.assertEqual(resp.sources[0].investigation_id, "BASS-II")
        self.assertIsNotNone(resp.sources[0].url)

    @patch("urllib.request.urlopen")
    def test_ollama_unavailable_fallback(self, mock_urlopen):
        """Test graceful fallback to deterministic template synthesis when Ollama is offline."""
        mock_urlopen.side_effect = urllib.error.URLError("[Errno 111] Connection refused")

        resp = self.assistant.answer("What is BASS-II?", use_llm=True)

        self.assertFalse(resp.llm_generated)
        self.assertIn("Ollama LLM was unavailable", resp.disclaimer)
        self.assertTrue(len(resp.answer) > 20)
        self.assertTrue(len(resp.sources) > 0)

    @patch("urllib.request.urlopen")
    def test_incorrect_model_name_404(self, mock_urlopen):
        """Test handling of 404 Model Not Found."""
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://127.0.0.1:11434/api/chat",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=io.BytesIO(b'{"error": "model not found"}'),
        )

        resp = self.assistant.answer("How do flames spread?", use_llm=True)

        self.assertFalse(resp.llm_generated)
        self.assertIn("Ollama LLM was unavailable", resp.disclaimer)

    @patch("urllib.request.urlopen")
    def test_request_timeout(self, mock_urlopen):
        """Test handling of server timeout."""
        mock_urlopen.side_effect = TimeoutError("Request timed out after 5s")

        resp = self.assistant.answer("Explain cool flames in microgravity", use_llm=True)

        self.assertFalse(resp.llm_generated)
        self.assertIn("timed out", resp.disclaimer.lower())
        self.assertTrue(len(resp.answer) > 20)

    @patch("urllib.request.urlopen")
    def test_empty_model_response(self, mock_urlopen):
        """Test handling of empty string returned by model."""
        mock_payload = {
            "message": {"role": "assistant", "content": "    "},
            "model": "llama3",
        }
        mock_urlopen.return_value = MockHTTPResponse(mock_payload)

        resp = self.assistant.answer("Explain fire suppression", use_llm=True)

        self.assertFalse(resp.llm_generated)
        self.assertIn("empty response", resp.disclaimer.lower())
        self.assertTrue(len(resp.answer) > 20)

    def test_no_relevant_retrieved_evidence(self):
        """Test abstention when no relevant combustion evidence exists."""
        # Unrelated query that matches 0 combustion chunks
        resp = self.assistant.answer("What is the recipe for chocolate chip cookies?", use_llm=True)

        self.assertEqual(resp.confidence, "insufficient_evidence")
        self.assertIn("insufficient evidence", resp.answer.lower())
        self.assertEqual(len(resp.sources), 0)

    @patch("urllib.request.urlopen")
    def test_invalid_citations(self, mock_urlopen):
        """Test that citations to non-retrieved or fabricated missions are rejected."""
        # Model claims evidence from an unretrieved / fabricated mission [APOLLO-99] alongside real [BASS-II]
        mock_payload = {
            "message": {
                "role": "assistant",
                "content": "According to [APOLLO-99] and [BASS-II], flame spread behaves differently in microgravity.",
            },
            "model": "llama3",
            "done": True,
        }
        mock_urlopen.return_value = MockHTTPResponse(mock_payload)

        resp = self.assistant.answer("Tell me about flame spread", use_llm=True)

        self.assertTrue(resp.llm_generated)
        self.assertIn("BASS-II", resp.validated_citations)
        self.assertIn("APOLLO-99", resp.rejected_citations)
        # Verify notice about rejected citation
        self.assertIn("[APOLLO-99]", resp.disclaimer)
        # Verify sources only contains the verified investigation
        source_ids = [s.investigation_id for s in resp.sources]
        self.assertIn("BASS-II", source_ids)
        self.assertNotIn("APOLLO-99", source_ids)

    @patch("urllib.request.urlopen")
    def test_multi_turn_conversation(self, mock_urlopen):
        """Test that conversational turns are sent to the LLM context."""
        captured_requests = []

        def fake_urlopen(req, timeout=None):
            captured_requests.append(req)
            return MockHTTPResponse({
                "message": {"role": "assistant", "content": "It used the Microgravity Science Glovebox [BASS-II]."},
                "model": "llama3",
            })

        mock_urlopen.side_effect = fake_urlopen

        history = [
            {"role": "user", "content": "What is BASS-II?"},
            {"role": "assistant", "content": "BASS-II is an ISS combustion experiment."},
        ]

        resp = self.assistant.answer(
            question="What facility did it use?",
            conversation_history=history,
            use_llm=True,
        )

        self.assertTrue(resp.llm_generated)
        self.assertEqual(len(captured_requests), 1)

        sent_payload = json.loads(captured_requests[0].data.decode("utf-8"))
        messages = sent_payload.get("messages", [])

        # Check messages include system prompt with evidence, previous user, previous assistant, and current user
        roles = [m["role"] for m in messages]
        self.assertEqual(roles[0], "system")
        self.assertEqual(roles[1], "user")
        self.assertEqual(roles[2], "assistant")
        self.assertEqual(roles[3], "user")
        self.assertEqual(messages[3]["content"], "What facility did it use?")

    def test_citation_links(self):
        """Test that generated citations contain valid URLs, DOIs, and active versions."""
        resp = self.assistant.answer("What is BASS-II?", use_llm=False)
        self.assertTrue(len(resp.sources) > 0)
        bass_citation = next((s for s in resp.sources if s.investigation_id == "BASS-II"), None)
        self.assertIsNotNone(bass_citation)
        self.assertEqual(bass_citation.active_version, 4)
        self.assertEqual(bass_citation.doi, "10.60555/4qc4-de67")
        self.assertTrue(bass_citation.url.startswith("http"))


if __name__ == "__main__":
    unittest.main()
