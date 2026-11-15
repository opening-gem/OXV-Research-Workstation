"""
Legacy BASS-II Experimental Archive Adapter.

Connects the legacy Burning and Suppression of Solids - II experimental processing pipeline
to the modern Flame in Freefall research catalogue.
Provides data provenance linkage to DOI 10.60555/4qc4-de67 and Version 4 dataset pointer.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.catalog.repository import ResearchCatalog
from src.catalog.schema import Investigation, InvestigationVersion


class BassIIAdapter:
    """
    Adapter bridging historical BASS-II flame spread calculations with the central PSI catalogue.
    """

    INVESTIGATION_ID = "BASS-II"
    CANONICAL_DOI = "10.60555/4qc4-de67"
    CANONICAL_VERSION = 4

    def __init__(self, catalog: Optional[ResearchCatalog] = None):
        self.catalog = catalog or ResearchCatalog()

    def get_canonical_record(self) -> Optional[Investigation]:
        """Fetch the central catalogue record for BASS-II."""
        return self.catalog.get(self.INVESTIGATION_ID)

    def verify_provenance_linkage(self) -> Dict[str, Any]:
        """Verify that the active catalogue entry maintains exact BASS-II DOI and Version 4."""
        record = self.get_canonical_record()
        if not record:
            return {"status": "error", "message": "BASS-II record not found in catalogue."}

        active_version = record.get_active_version()
        is_doi_matched = (active_version.doi == self.CANONICAL_DOI) if active_version else False
        is_ver_matched = (record.active_version == self.CANONICAL_VERSION)

        return {
            "investigation_id": self.INVESTIGATION_ID,
            "status": "verified" if (is_doi_matched and is_ver_matched) else "mismatch",
            "active_version": record.active_version,
            "expected_version": self.CANONICAL_VERSION,
            "active_doi": active_version.doi if active_version else None,
            "expected_doi": self.CANONICAL_DOI,
            "doi_matches": is_doi_matched,
            "version_matches": is_ver_matched,
            "source_url": record.source_url,
            "psi_url": record.psi_url,
        }

    def format_experimental_citation(self) -> str:
        """Generate standardized NASA scientific citation for BASS-II."""
        record = self.get_canonical_record()
        if not record:
            return "Ferkul, P. et al. Burning and Suppression of Solids - II (BASS-II). NASA PSI Repository."
        active_ver = record.get_active_version()
        doi_str = f"https://doi.org/{active_ver.doi}" if active_ver and active_ver.doi else "NASA PSI"
        return (
            f"Ferkul, P., Olson, S., et al. ({record.start_date.year if record.start_date else 2012}). "
            f"Burning and Suppression of Solids - II (BASS-II), Version {record.active_version}. "
            f"NASA Physical Sciences Informatics System. DOI: {doi_str}"
        )
