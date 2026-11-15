"""Unit tests for NASA data provenance, version tracking, and BASS-II legacy adapter."""
import unittest

from legacy.bass_ii.adapters import BassIIAdapter
from legacy.bass_ii.analysis import BassIIAnalysis
from src.catalog.repository import ResearchCatalog
from src.ingestion.version_tracker import VersionTracker


class TestProvenanceAndLegacyAdapter(unittest.TestCase):
    def test_bass_ii_adapter_and_doi(self):
        adapter = BassIIAdapter()
        linkage = adapter.verify_provenance_linkage()
        self.assertEqual(linkage["status"], "verified")
        self.assertEqual(linkage["active_version"], 4)
        self.assertEqual(linkage["active_doi"], "10.60555/4qc4-de67")
        self.assertTrue(linkage["doi_matches"])
        self.assertTrue(linkage["version_matches"])

        citation = adapter.format_experimental_citation()
        self.assertIn("10.60555/4qc4-de67", citation)
        self.assertIn("BASS-II", citation)

    def test_bass_ii_analysis_reproducibility(self):
        analysis = BassIIAnalysis()
        out = analysis.run_flame_spread_analysis()
        self.assertIn("peak_microgravity_rate_mm_s", out)
        self.assertEqual(out["doi"], "10.60555/4qc4-de67")
        self.assertTrue(out["peak_microgravity_rate_mm_s"] > 1.0)

    def test_version_tracker_audit_log(self):
        catalog = ResearchCatalog()
        bass_record = catalog.get("BASS-II")
        self.assertIsNotNone(bass_record)

        tracker = VersionTracker()
        report = tracker.verify_provenance(bass_record)
        self.assertTrue(report["has_doi"])
        self.assertEqual(report["active_doi"], "10.60555/4qc4-de67")
        self.assertTrue(report["is_fully_traceable"])


if __name__ == "__main__":
    unittest.main()
