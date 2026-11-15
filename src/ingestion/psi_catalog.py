"""
NASA PSI Investigation Catalogue Ingester.

Loads canonical investigation records from the Physical Sciences Informatics seed repository,
normalizes and validates all schema fields, maintains version trees, and emits ingestion manifests.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.catalog.normalizer import InvestigationNormalizer
from src.catalog.repository import ResearchCatalog
from src.catalog.schema import (
    Investigation,
    InvestigationStatus,
    InvestigationVersion,
    Investigator,
    ResearchCategory,
    SpacecraftPlatform,
)
from src.config import ProjectConfig
from src.ingestion.version_tracker import VersionTracker

logger = logging.getLogger(__name__)


class PSICatalogIngester:
    """
    Ingests and normalizes NASA combustion investigations into the persistent catalogue.
    """

    def __init__(self, config: Optional[ProjectConfig] = None):
        self.config = config or ProjectConfig()
        self.normalizer = InvestigationNormalizer()
        self.version_tracker = VersionTracker(
            self.config.manifests_dir / "version_history.json"
        )
        self.catalog = ResearchCatalog(self.config.catalog_path)

    def ingest_from_seed(self, seed_path: Optional[Path | str] = None) -> List[Investigation]:
        """Load, validate, normalize, and persist investigations from seed JSON file."""
        path = Path(seed_path) if seed_path else self.config.raw_psi_dir / "seed_investigations.json"
        if not path.exists():
            raise FileNotFoundError(f"Seed file not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_records = json.load(f)

        logger.info("Ingesting %d investigations from %s", len(raw_records), path)
        investigations: List[Investigation] = []

        for raw in raw_records:
            try:
                # Build investigators
                investigators_list = []
                for p in raw.get("investigators", []):
                    investigators_list.append(
                        Investigator(
                            name=self.normalizer.clean_text(p.get("name", "Unknown")),
                            affiliation=p.get("affiliation"),
                            role=p.get("role", "Researcher"),
                        )
                    )

                # Build categories
                cats = []
                for c in raw.get("categories", []):
                    resolved = ResearchCategory.from_string(c)
                    if resolved not in cats:
                        cats.append(resolved)

                # Build version
                version_num = int(raw.get("version", 1))
                doi = raw.get("doi")
                start_year = raw.get("start_year")
                start_dt = date(int(start_year), 1, 1) if start_year else None
                end_year = raw.get("end_year")
                end_dt = date(int(end_year), 12, 31) if end_year else None

                v_desc = f"Initial catalogue entry for version {version_num}."
                version_obj = InvestigationVersion(
                    version_number=version_num,
                    doi=doi,
                    release_date=start_dt,
                    description=v_desc,
                    is_active=True,
                    retrieved_at=datetime.now(timezone.utc),
                )

                # Platform and status
                plat_str = str(raw.get("platform", "iss")).lower()
                platform = (
                    SpacecraftPlatform.CYGNUS if "cygnus" in plat_str
                    else SpacecraftPlatform.ISS if (re.search(r"\biss\b", plat_str) or plat_str == "iss")
                    else SpacecraftPlatform.DROP_TOWER if "drop" in plat_str
                    else SpacecraftPlatform.PARABOLIC_FLIGHT if "parabolic" in plat_str
                    else SpacecraftPlatform.SOUNDING_ROCKET if "rocket" in plat_str
                    else SpacecraftPlatform.GROUND if "ground" in plat_str
                    else SpacecraftPlatform.OTHER
                )

                stat_str = str(raw.get("status", "completed")).lower()
                status = (
                    InvestigationStatus.COMPLETED if "completed" in stat_str
                    else InvestigationStatus.ACTIVE if "active" in stat_str
                    else InvestigationStatus.PLANNED if "planned" in stat_str
                    else InvestigationStatus.ARCHIVED
                )

                inv = Investigation(
                    id=self.normalizer.normalize_id(raw.get("id", "UNKNOWN")),
                    title=self.normalizer.clean_text(raw.get("title", "")),
                    full_name=self.normalizer.clean_text(raw.get("full_name", "")),
                    description=self.normalizer.clean_text(raw.get("description", "")),
                    objectives=[self.normalizer.clean_text(o) for o in raw.get("objectives", []) if o],
                    categories=cats,
                    status=status,
                    platform=platform,
                    facility=raw.get("facility"),
                    start_date=start_dt,
                    end_date=end_dt,
                    investigators=investigators_list,
                    source_url=raw.get("source_url", "https://psi.nasa.gov"),
                    psi_url=raw.get("psi_url"),
                    versions=[version_obj],
                    active_version=version_num,
                    keywords=raw.get("keywords", []),
                    related_investigations=raw.get("related_investigations", []),
                    applications=raw.get("applications", []),
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )

                investigations.append(inv)
                self.version_tracker.track_update(
                    investigation_id=inv.id,
                    new_version=version_num,
                    doi=doi,
                    description=v_desc,
                )
            except Exception as e:
                logger.error("Failed to parse investigation record %s: %s", raw.get("id"), e)

        # Upsert into catalog
        for inv in investigations:
            try:
                self.catalog.add(inv)
            except ValueError:
                # If already exists, update
                self.catalog.update(inv)

        # Write manifest
        self._write_manifest(investigations)
        logger.info("Successfully ingested %d investigations into catalogue", len(investigations))
        return investigations

    def _write_manifest(self, investigations: List[Investigation]) -> Dict[str, Any]:
        """Generate and save data ingestion manifest."""
        manifest_path = self.config.manifests_dir / "ingestion_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        completeness_scores = [self._calculate_completeness(inv) for inv in investigations]
        avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0

        manifest = {
            "last_ingestion": datetime.now(timezone.utc).isoformat(),
            "source": "seed_repository",
            "investigations_count": len(investigations),
            "average_completeness_score": round(avg_completeness, 3),
            "investigation_ids": [inv.id for inv in investigations],
            "categories_found": sorted(
                list({c.value for inv in investigations for c in inv.categories})
            ),
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def _calculate_completeness(self, inv: Investigation) -> float:
        """Measure completeness of fields in an investigation record."""
        total_checks = 10
        score = 0
        if inv.title: score += 1
        if inv.full_name: score += 1
        if len(inv.description) > 50: score += 1
        if len(inv.objectives) > 0: score += 1
        if len(inv.categories) > 0: score += 1
        if inv.facility: score += 1
        if len(inv.investigators) > 0: score += 1
        if inv.source_url: score += 1
        if inv.get_active_version() and inv.get_active_version().doi: score += 1
        if len(inv.applications) > 0: score += 1
        return score / total_checks

    def validate_completeness(self, investigations: List[Investigation]) -> Dict[str, Any]:
        """Report on data completeness across all records."""
        scores = {inv.id: self._calculate_completeness(inv) for inv in investigations}
        return {
            "per_investigation": scores,
            "mean_completeness": sum(scores.values()) / len(scores) if scores else 0.0,
        }
