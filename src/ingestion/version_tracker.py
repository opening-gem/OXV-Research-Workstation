"""
Dataset Version Tracking and Provenance Management.

Tracks the lifecycle of NASA PSI datasets, DOI assignments, version increments,
and audit logs to ensure scientific reproducibility and attribution.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.catalog.schema import Investigation, InvestigationVersion

logger = logging.getLogger(__name__)


class VersionTracker:
    """
    Manages versioning and change audit trail for investigations.
    Maintains persistent record of releases, DOIs, and metadata transitions.
    """

    def __init__(self, history_file: Path | str | None = None):
        self.history_file = (
            Path(history_file)
            if history_file
            else Path(__file__).resolve().parents[2] / "data" / "manifests" / "version_history.json"
        )
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_file.exists():
            self._save_history({})

    def _load_history(self) -> Dict[str, Any]:
        if not self.history_file.exists():
            return {}
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error("Corrupted history file %s: %s", self.history_file, e)
            try:
                import shutil
                bak = self.history_file.with_suffix(".corrupt.bak")
                shutil.copy2(self.history_file, bak)
                logger.warning("Backed up corrupted history file to %s", bak)
            except Exception:
                pass
            return {}
        except Exception as e:
            logger.error("Failed to read history file %s: %s", self.history_file, e)
            return {}

    def _save_history(self, history: Dict[str, Any]) -> None:
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)

    def track_update(
        self,
        investigation_id: str,
        new_version: int,
        doi: Optional[str],
        description: str,
        changed_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Record a version update event in the audit log if state has changed."""
        history = self._load_history()
        events = history.setdefault(investigation_id, [])

        if events:
            latest = events[-1]
            if latest.get("version_number") == new_version and latest.get("doi") == doi:
                logger.debug("Skipping duplicate version %d event for %s", new_version, investigation_id)
                return latest

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version_number": new_version,
            "doi": doi,
            "description": description,
            "changed_fields": changed_fields or [],
        }
        events.append(event)
        self._save_history(history)
        logger.info("Recorded version %d for %s (DOI: %s)", new_version, investigation_id, doi)
        return event

    def get_version_history(self, investigation_id: str) -> List[Dict[str, Any]]:
        """Retrieve the historical changelog for an investigation."""
        history = self._load_history()
        return history.get(investigation_id, [])

    def verify_provenance(self, investigation: Investigation) -> Dict[str, Any]:
        """Check whether an investigation meets NASA provenance and attribution guidelines."""
        active_ver = investigation.get_active_version()
        has_doi = bool(active_ver and active_ver.doi)
        has_psi_link = bool(investigation.psi_url)
        has_source = bool(investigation.source_url)
        version_count = len(investigation.versions)

        score = 0.0
        if has_doi:
            score += 0.4
        if has_psi_link:
            score += 0.3
        if has_source:
            score += 0.2
        if version_count > 0:
            score += 0.1

        return {
            "investigation_id": investigation.id,
            "provenance_score": round(score, 2),
            "has_doi": has_doi,
            "active_doi": active_ver.doi if active_ver else None,
            "has_psi_url": has_psi_link,
            "version_count": version_count,
            "active_version": investigation.active_version,
            "is_fully_traceable": score >= 0.8,
        }
