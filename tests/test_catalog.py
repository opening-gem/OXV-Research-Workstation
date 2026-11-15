"""Unit tests for catalog schema and repository."""
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from src.catalog.repository import ResearchCatalog
from src.catalog.schema import (
    Investigation,
    InvestigationStatus,
    InvestigationVersion,
    Investigator,
    ResearchCategory,
    SpacecraftPlatform,
)


class TestCatalogSchemaAndRepository(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.catalog_file = Path(self.temp_dir.name) / "test_catalog.json"
        self.catalog = ResearchCatalog(self.catalog_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_investigation_creation_and_serialization(self):
        ver = InvestigationVersion(
            version_number=1,
            doi="10.60555/test-0001",
            release_date=date(2023, 1, 1),
            description="Initial test version",
            is_active=True,
            retrieved_at=datetime.now(timezone.utc),
        )
        inv = Investigation(
            id="TEST-EXP",
            title="Test Microgravity Experiment",
            full_name="Testing Microgravity Combustion Fundamentals",
            description="Experimental test investigation on soot dynamics in spaceflight.",
            objectives=["Quantify soot volume fraction in microgravity"],
            categories=[ResearchCategory.SOOT_FORMATION],
            status=InvestigationStatus.COMPLETED,
            platform=SpacecraftPlatform.ISS,
            facility="Combustion Integrated Rack",
            investigators=[Investigator(name="Dr. Test", affiliation="NASA GRC", role="PI")],
            source_url="https://psi.nasa.gov",
            versions=[ver],
            active_version=1,
            keywords=["soot", "iss", "microgravity"],
            related_investigations=[],
            applications=["Spacecraft fire safety"],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        self.assertEqual(inv.id, "TEST-EXP")
        self.assertEqual(inv.active_version, 1)
        self.assertEqual(inv.get_active_version().doi, "10.60555/test-0001")

        # Test model_dump and model_validate
        data = inv.model_dump(mode="json")
        self.assertEqual(data["id"], "TEST-EXP")
        restored = Investigation.model_validate(data)
        self.assertEqual(restored.id, "TEST-EXP")
        self.assertEqual(len(restored.versions), 1)

    def test_repository_crud_operations(self):
        ver = InvestigationVersion(
            version_number=1,
            doi="10.60555/test-0002",
            description="Version 1",
            is_active=True,
        )
        inv = Investigation(
            id="EXP-1",
            title="Experiment One",
            full_name="Experiment One Full",
            description="Description for experiment one.",
            objectives=["Obj 1"],
            categories=[ResearchCategory.DROPLET_COMBUSTION],
            status=InvestigationStatus.ACTIVE,
            platform=SpacecraftPlatform.DROP_TOWER,
            source_url="https://psi.nasa.gov",
            versions=[ver],
            active_version=1,
        )

        # Add
        self.catalog.add(inv)
        self.assertEqual(len(self.catalog.list_all()), 1)

        # Get
        fetched = self.catalog.get("EXP-1")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.title, "Experiment One")

        # Update
        fetched.title = "Updated Title One"
        self.catalog.update(fetched)
        updated = self.catalog.get("EXP-1")
        self.assertEqual(updated.title, "Updated Title One")

        # Search
        matches = self.catalog.full_text_search("experiment")
        self.assertEqual(len(matches), 1)

        # Delete
        deleted = self.catalog.delete("EXP-1")
        self.assertTrue(deleted)
        self.assertEqual(len(self.catalog.list_all()), 0)


if __name__ == "__main__":
    unittest.main()
