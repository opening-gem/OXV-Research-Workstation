"""Unit tests for AI subsystems: TF-IDF, classifier, ranker, and evaluator."""
import unittest
import numpy as np

from src.ai.compat_ml import (
    TfidfVectorizer,
    OneVsRestLogisticRegression,
    MultiLabelBinarizer,
    cosine_similarity,
)
from src.ai.classifier import ResearchClassifier
from src.ai.embeddings import EmbeddingsManager
from src.ai.ranker import ResearchRanker
from src.ai.evaluation import AIEvaluator, precision_at_k, ndcg_at_k


class TestAISubsystems(unittest.TestCase):
    def test_tfidf_and_cosine_similarity(self):
        docs = [
            "spacecraft fire safety in microgravity",
            "droplet combustion and flame extinction",
            "smoke detection aerosol sensor on space station",
        ]
        vec = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
        X = vec.fit_transform(docs)
        self.assertEqual(X.shape[0], 3)
        self.assertTrue(X.shape[1] > 0)

        # Check cosine similarity
        sim = cosine_similarity(X)
        self.assertEqual(sim.shape, (3, 3))
        # Self-similarity should be 1.0
        for i in range(3):
            self.assertAlmostEqual(sim[i, i], 1.0, places=4)

    def test_classifier_training_and_explanation(self):
        data = [
            {"investigation_id": "1", "text": "flame spread on solid materials fire safety", "categories": ["spacecraft_fire_safety", "flame_spread"]},
            {"investigation_id": "2", "text": "droplet combustion and extinction", "categories": ["droplet_combustion"]},
            {"investigation_id": "3", "text": "smoke detection and aerosol measurement", "categories": ["smoke_detection"]},
            {"investigation_id": "4", "text": "solid fuel burning and flammability fire safety", "categories": ["spacecraft_fire_safety", "material_flammability"]},
        ]
        clf = ResearchClassifier()
        metrics = clf.train(data)
        self.assertIn("macro_f1", metrics)
        self.assertTrue(clf.is_trained)

        # Predict
        preds = clf.predict("flame spread over solid sample")
        self.assertTrue(len(preds) > 0)

        # Explain
        exp = clf.explain_prediction("flame spread over solid sample")
        self.assertIn("predicted_categories", exp)
        self.assertIn("explanations", exp)

    def test_ranker_and_ir_metrics(self):
        em = EmbeddingsManager()
        em.fit(["fire safety flame spread", "droplet combustion extinction"])
        ranker = ResearchRanker(embeddings=em)

        invs = [
            {"id": "A", "title": "Fire Study", "text": "fire safety and flame spread on ISS", "objectives": ["Study fire safety"]},
            {"id": "B", "title": "Droplet Study", "text": "droplet combustion in drop tower", "objectives": ["Study droplets"]},
        ]
        ranked = ranker.rank("fire safety", invs, top_k=2)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].investigation_id, "A")
        self.assertTrue(ranked[0].overall_score >= ranked[1].overall_score)

        # NDCG@k and Precision@k checks
        retrieved = ["A", "B"]
        relevant = {"A"}
        grades = {"A": 3, "B": 0}
        p_at_1 = precision_at_k(retrieved, relevant, k=1)
        self.assertEqual(p_at_1, 1.0)
        ndcg_val = ndcg_at_k(retrieved, grades, k=2)
        self.assertAlmostEqual(ndcg_val, 1.0, places=3)

    def test_rag_evaluation(self):
        from src.rag.chunker import DocumentChunker
        from src.rag.retriever import InvestigationRetriever
        from src.rag.assistant import ResearchAssistant
        from src.catalog.repository import ResearchCatalog

        catalog = ResearchCatalog()
        invs = [inv.model_dump() for inv in catalog.list_all()]
        chunker = DocumentChunker()
        chunks = chunker.chunk_all(invs)
        retriever = InvestigationRetriever()
        retriever.build_index(chunks)
        assistant = ResearchAssistant(retriever=retriever, catalog=catalog)

        evaluator = AIEvaluator()
        queries = [
            {"query": "fire safety in microgravity", "relevant_investigations": ["BASS-II", "SAFFIRE-I"]},
            {"query": "microwave cooking steak", "relevant_investigations": [], "is_out_of_domain": True},
        ]
        rag_report = evaluator.evaluate_rag(assistant, queries, use_llm=False)
        self.assertEqual(rag_report.n_queries, 2)
        self.assertAlmostEqual(rag_report.citation_validity_rate, 1.0)
        self.assertTrue(rag_report.abstention_accuracy >= 0.5)
        self.assertIn("PRELIMINARY", rag_report.benchmark_status)

        # Check full text report generation
        full_report = evaluator.generate_report()
        self.assertIn("RAG & Grounding", full_report)


if __name__ == "__main__":
    unittest.main()
