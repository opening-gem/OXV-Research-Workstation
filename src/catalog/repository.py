"""
JSON-backed repository for the investigation catalogue.

Provides thread-safe CRUD operations, search, version tracking, and
aggregate statistics.  Data is persisted to a single JSON file with
file-level locking so that concurrent processes (e.g. the ingestion
pipeline and the Streamlit dashboard) do not corrupt the catalogue.

Typical usage::

    from src.catalog.repository import ResearchCatalog

    catalog = ResearchCatalog()
    catalog.add(investigation)
    results = catalog.search_by_category(ResearchCategory.COOL_FLAMES)
    stats = catalog.get_statistics()
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:
    fcntl = None  # Windows fallback

from src.catalog.schema import (
    Investigation,
    InvestigationStatus,
    InvestigationVersion,
    ResearchCategory,
    SpacecraftPlatform,
)
from src.config import ProjectConfig

logger = logging.getLogger(__name__)


class ResearchCatalog:
    """Thread-safe, JSON-backed repository for combustion investigations.

    The catalogue file (``catalog.json``) is a JSON array of serialised
    :class:`Investigation` objects.  Every public mutation method acquires
    an exclusive file lock before reading/writing.

    Args:
        path: Explicit path to the catalogue JSON file.  When *None*,
              falls back to the project configuration default.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        if path is not None:
            self._path = Path(path)
        else:
            self._path = ProjectConfig().catalog_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("[]", encoding="utf-8")

    # ------------------------------------------------------------------
    # Internal I/O helpers (all acquire / release the file lock)
    # ------------------------------------------------------------------

    def _read_locked(self) -> list[Investigation]:
        """Read the catalogue, acquiring a shared lock when supported."""
        with open(self._path, "r", encoding="utf-8") as fh:
            if fcntl:
                fcntl.flock(fh, fcntl.LOCK_SH)
            try:
                raw: list[dict[str, Any]] = json.load(fh)
            finally:
                if fcntl:
                    fcntl.flock(fh, fcntl.LOCK_UN)
        return [Investigation.model_validate(obj) for obj in raw]

    def _write_locked(self, investigations: list[Investigation]) -> None:
        """Persist catalogue atomically under exclusive lock when supported."""
        payload = [
            inv.model_dump(mode="json") for inv in investigations
        ]
        tmp_path = self._path.with_name(f"{self._path.name}.tmp.{os.getpid()}")
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                if fcntl:
                    fcntl.flock(fh, fcntl.LOCK_EX)
                try:
                    json.dump(payload, fh, indent=2, default=str)
                    fh.flush()
                    os.fsync(fh.fileno())
                finally:
                    if fcntl:
                        fcntl.flock(fh, fcntl.LOCK_UN)
            os.replace(tmp_path, self._path)
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, investigation: Investigation) -> Investigation:
        """Add a new investigation to the catalogue.

        Args:
            investigation: A fully validated :class:`Investigation` instance.

        Returns:
            The investigation as stored (with any auto-set fields).

        Raises:
            ValueError: If an investigation with the same ``id`` already exists.
        """
        investigations = self._read_locked()
        existing_ids = {inv.id for inv in investigations}
        if investigation.id in existing_ids:
            raise ValueError(
                f"Investigation '{investigation.id}' already exists in the catalogue."
            )
        investigation.created_at = datetime.now(timezone.utc)
        investigation.updated_at = datetime.now(timezone.utc)
        investigations.append(investigation)
        self._write_locked(investigations)
        logger.info("Added investigation '%s'.", investigation.id)
        return investigation

    def get(self, investigation_id: str) -> Investigation | None:
        """Retrieve a single investigation by its ID.

        Args:
            investigation_id: Case-insensitive identifier.

        Returns:
            The matching :class:`Investigation`, or ``None``.
        """
        normalised = investigation_id.strip().upper()
        for inv in self._read_locked():
            if inv.id == normalised:
                return inv
        return None

    def list_all(self) -> list[Investigation]:
        """Return every investigation in the catalogue.

        Returns:
            List of all :class:`Investigation` objects (may be empty).
        """
        return self._read_locked()

    def update(self, investigation: Investigation) -> Investigation:
        """Replace an existing investigation record.

        The record is matched by ``investigation.id``.  The ``updated_at``
        timestamp is refreshed automatically.

        Args:
            investigation: Updated investigation data.

        Returns:
            The investigation as stored.

        Raises:
            KeyError: If no investigation with a matching ``id`` is found.
        """
        investigations = self._read_locked()
        for idx, inv in enumerate(investigations):
            if inv.id == investigation.id:
                investigation.updated_at = datetime.now(timezone.utc)
                investigations[idx] = investigation
                self._write_locked(investigations)
                logger.info("Updated investigation '%s'.", investigation.id)
                return investigation
        raise KeyError(
            f"Investigation '{investigation.id}' not found in the catalogue."
        )

    def upsert(self, investigation: Investigation) -> Investigation:
        """Insert or update an investigation.

        If the ``id`` already exists the record is replaced; otherwise a
        new record is created.

        Args:
            investigation: Investigation data.

        Returns:
            The investigation as stored.
        """
        try:
            return self.update(investigation)
        except KeyError:
            return self.add(investigation)

    def delete(self, investigation_id: str) -> bool:
        """Remove an investigation from the catalogue.

        Args:
            investigation_id: Identifier of the investigation to remove.

        Returns:
            ``True`` if a record was deleted, ``False`` if not found.
        """
        normalised = investigation_id.strip().upper()
        investigations = self._read_locked()
        filtered = [inv for inv in investigations if inv.id != normalised]
        if len(filtered) == len(investigations):
            return False
        self._write_locked(filtered)
        logger.info("Deleted investigation '%s'.", normalised)
        return True

    def count(self) -> int:
        """Return the total number of investigations.

        Returns:
            Non-negative integer count.
        """
        return len(self._read_locked())

    # ------------------------------------------------------------------
    # Version tracking
    # ------------------------------------------------------------------

    def add_version(
        self,
        investigation_id: str,
        version: InvestigationVersion,
    ) -> Investigation:
        """Append a dataset version to an existing investigation.

        The new version is marked as active and all prior versions are
        deactivated.  The investigation's ``active_version`` pointer is
        advanced accordingly.

        Args:
            investigation_id: Target investigation.
            version: The new version to append.

        Returns:
            The updated :class:`Investigation`.

        Raises:
            KeyError: If the investigation is not found.
            ValueError: If the version number already exists.
        """
        inv = self.get(investigation_id)
        if inv is None:
            raise KeyError(
                f"Investigation '{investigation_id}' not found in the catalogue."
            )
        inv.add_version(version)
        return self.update(inv)

    # ------------------------------------------------------------------
    # Search / query
    # ------------------------------------------------------------------

    def search_by_id(self, pattern: str) -> list[Investigation]:
        """Search investigations whose ID contains *pattern* (case-insensitive).

        Args:
            pattern: Substring to match against investigation IDs.

        Returns:
            List of matching investigations.
        """
        normalised = pattern.strip().upper()
        return [
            inv for inv in self._read_locked() if normalised in inv.id
        ]

    def search_by_category(
        self, category: ResearchCategory
    ) -> list[Investigation]:
        """Return investigations tagged with a given category.

        Both human-assigned (``categories``) and ML-predicted
        (``predicted_categories``) labels are searched.

        Args:
            category: The :class:`ResearchCategory` to filter by.

        Returns:
            List of matching investigations.
        """
        results: list[Investigation] = []
        for inv in self._read_locked():
            if category in inv.categories:
                results.append(inv)
            elif inv.predicted_categories and category in inv.predicted_categories:
                results.append(inv)
        return results

    def search_by_keyword(self, keyword: str) -> list[Investigation]:
        """Return investigations whose keywords contain *keyword*.

        The match is case-insensitive and substring-based so that
        ``"flame"`` matches ``"cool flames"``.

        Args:
            keyword: Search term.

        Returns:
            List of matching investigations.
        """
        kw_lower = keyword.strip().lower()
        return [
            inv
            for inv in self._read_locked()
            if any(kw_lower in k for k in inv.keywords)
        ]

    def search_by_status(
        self, status: InvestigationStatus
    ) -> list[Investigation]:
        """Return investigations with the given lifecycle status.

        Args:
            status: Target :class:`InvestigationStatus`.

        Returns:
            List of matching investigations.
        """
        return [
            inv for inv in self._read_locked() if inv.status == status
        ]

    def search_by_platform(
        self, platform: SpacecraftPlatform
    ) -> list[Investigation]:
        """Return investigations conducted on a given platform.

        Args:
            platform: Target :class:`SpacecraftPlatform`.

        Returns:
            List of matching investigations.
        """
        return [
            inv for inv in self._read_locked() if inv.platform == platform
        ]

    def full_text_search(self, query: str) -> list[Investigation]:
        """Simple full-text search across all textual fields.

        Searches the concatenated searchable text (via
        :meth:`Investigation.to_search_text`) for the query string,
        case-insensitively.

        Args:
            query: Free-text search query.

        Returns:
            List of matching investigations, ordered by occurrence count
            (descending).
        """
        q_lower = query.strip().lower()
        if not q_lower:
            return []

        scored: list[tuple[int, Investigation]] = []
        for inv in self._read_locked():
            text = inv.to_search_text().lower()
            occurrences = text.count(q_lower)
            if occurrences > 0:
                scored.append((occurrences, inv))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [inv for _, inv in scored]

    # ------------------------------------------------------------------
    # Aggregates / statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """Compute aggregate statistics over the entire catalogue.

        Returns:
            A dictionary with the following keys:

            - ``total``: Total investigation count.
            - ``by_category``: ``{category_value: count}``.
            - ``by_platform``: ``{platform_value: count}``.
            - ``by_status``: ``{status_value: count}``.
            - ``with_embeddings``: Number of investigations that have
              embeddings.
            - ``with_versions``: Number with at least one dataset version.
            - ``total_versions``: Sum of all versions across investigations.
            - ``top_keywords``: 20 most common keywords as
              ``[(keyword, count), ...]``.
        """
        investigations = self._read_locked()

        category_counter: Counter[str] = Counter()
        platform_counter: Counter[str] = Counter()
        status_counter: Counter[str] = Counter()
        keyword_counter: Counter[str] = Counter()
        with_embeddings = 0
        with_versions = 0
        total_versions = 0

        for inv in investigations:
            for cat in inv.categories:
                category_counter[cat.value] += 1
            platform_counter[inv.platform.value] += 1
            status_counter[inv.status.value] += 1
            keyword_counter.update(inv.keywords)
            if inv.has_embeddings:
                with_embeddings += 1
            if inv.versions:
                with_versions += 1
                total_versions += len(inv.versions)

        return {
            "total": len(investigations),
            "by_category": dict(category_counter),
            "by_platform": dict(platform_counter),
            "by_status": dict(status_counter),
            "with_embeddings": with_embeddings,
            "with_versions": with_versions,
            "total_versions": total_versions,
            "top_keywords": keyword_counter.most_common(20),
        }
