"""
Metadata normalisation utilities for combustion research investigations.

Cleans, standardises, and deduplicates ingested investigation metadata
before it is persisted to the catalogue.

Typical usage::

    from src.catalog.normalizer import InvestigationNormalizer

    normalizer = InvestigationNormalizer()
    clean = normalizer.normalize(raw_dict)
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import urlparse

from src.catalog.schema import (
    Investigation,
    InvestigationVersion,
    Investigator,
    InvestigationStatus,
    ResearchCategory,
    SpacecraftPlatform,
)

logger = logging.getLogger(__name__)


class InvestigationNormalizer:
    """Clean and standardise raw investigation metadata dictionaries.

    The normaliser is intentionally stateless — every method is a pure
    function of its input.  It is safe to share a single instance across
    threads.
    """

    # Characters that are never meaningful in our text fields.
    _CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

    # Whitespace runs (including non-breaking spaces).
    _WHITESPACE_RE = re.compile(r"[\s\u00a0]+")

    # Common date formats encountered in NASA metadata.
    _DATE_FORMATS: list[str] = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d %b %Y",
        "%d %B %Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%Y/%m/%d",
        "%Y%m%d",
        "%Y",
    ]

    # -----------------------------------------------------------------------
    # Text cleaning
    # -----------------------------------------------------------------------

    def normalize_id(self, text: str | None) -> str:
        """Normalise an investigation ID."""
        return self.clean_text(text).upper()

    def clean_text(self, text: str | None) -> str:
        """Normalise a free-text field.

        Steps:
        1. Unicode NFC normalisation.
        2. Strip control characters.
        3. Collapse whitespace runs to a single space.
        4. Strip leading/trailing whitespace.

        Args:
            text: Raw text value.

        Returns:
            Cleaned string (may be empty).
        """
        if not text:
            return ""
        text = unicodedata.normalize("NFC", text)
        text = self._CONTROL_CHAR_RE.sub("", text)
        text = self._WHITESPACE_RE.sub(" ", text)
        return text.strip()

    def clean_multiline(self, text: str | None) -> str:
        """Clean a multi-line text field, preserving intentional line breaks.

        Each line is individually cleaned, blank lines are collapsed, and
        trailing whitespace is removed.

        Args:
            text: Raw multi-line text.

        Returns:
            Cleaned string.
        """
        if not text:
            return ""
        lines = text.splitlines()
        cleaned = [self.clean_text(line) for line in lines]
        # Collapse consecutive blank lines into one.
        result: list[str] = []
        prev_blank = False
        for line in cleaned:
            if not line:
                if not prev_blank:
                    result.append("")
                prev_blank = True
            else:
                result.append(line)
                prev_blank = False
        return "\n".join(result).strip()

    # -----------------------------------------------------------------------
    # Date normalisation
    # -----------------------------------------------------------------------

    def parse_date(self, value: str | date | None) -> date | None:
        """Parse a date value from a variety of common formats.

        Args:
            value: A string or :class:`date` object.

        Returns:
            A :class:`date` instance, or ``None`` if parsing fails.
        """
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        text = self.clean_text(str(value))
        if not text:
            return None
        for fmt in self._DATE_FORMATS:
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        logger.warning("Unable to parse date: '%s'", text)
        return None

    # -----------------------------------------------------------------------
    # URL validation
    # -----------------------------------------------------------------------

    def validate_url(self, url: str | None) -> str | None:
        """Validate and clean a URL.

        Returns ``None`` for empty or malformed values.

        Args:
            url: Raw URL string.

        Returns:
            Cleaned URL or ``None``.
        """
        if not url:
            return None
        url = self.clean_text(url)
        if not url:
            return None
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            logger.warning("Invalid URL scheme: '%s'", url)
            return None
        if not parsed.netloc:
            logger.warning("URL missing netloc: '%s'", url)
            return None
        return url

    # -----------------------------------------------------------------------
    # Keyword normalisation
    # -----------------------------------------------------------------------

    def normalize_keywords(self, keywords: list[str] | None) -> list[str]:
        """Lower-case, deduplicate, and sort a keyword list.

        Synonyms are collapsed (e.g. ``"ISS"`` → ``"international space station"``
        is *not* done here — that belongs in a domain thesaurus).  This method
        focuses on mechanical cleaning only.

        Args:
            keywords: Raw keyword list.

        Returns:
            Sorted, deduplicated list of lowercase keywords.
        """
        if not keywords:
            return []
        seen: set[str] = set()
        result: list[str] = []
        for kw in keywords:
            cleaned = self.clean_text(kw).lower()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
        result.sort()
        return result

    # -----------------------------------------------------------------------
    # Investigator normalisation
    # -----------------------------------------------------------------------

    def normalize_investigator(self, raw: dict[str, Any]) -> Investigator:
        """Build a clean :class:`Investigator` from a raw dictionary.

        Args:
            raw: Dictionary with at least a ``"name"`` key.

        Returns:
            A validated :class:`Investigator`.

        Raises:
            ValueError: If the name is empty after cleaning.
        """
        name = self.clean_text(raw.get("name", ""))
        if not name:
            raise ValueError("Investigator name cannot be empty.")
        affiliation = self.clean_text(raw.get("affiliation")) or None
        role = self.clean_text(raw.get("role")) or None
        return Investigator(name=name, affiliation=affiliation, role=role)

    # -----------------------------------------------------------------------
    # Version normalisation
    # -----------------------------------------------------------------------

    def normalize_version(self, raw: dict[str, Any]) -> InvestigationVersion:
        """Build a clean :class:`InvestigationVersion` from raw data.

        Args:
            raw: Dictionary with version fields.

        Returns:
            A validated :class:`InvestigationVersion`.
        """
        return InvestigationVersion(
            version_number=int(raw.get("version_number", 1)),
            doi=self.clean_text(raw.get("doi")) or None,
            release_date=self.parse_date(raw.get("release_date")),
            description=self.clean_text(raw.get("description", "Initial release")),
            is_active=bool(raw.get("is_active", False)),
            retrieved_at=datetime.now(timezone.utc),
        )

    # -----------------------------------------------------------------------
    # Full investigation normalisation
    # -----------------------------------------------------------------------

    def normalize(self, raw: dict[str, Any]) -> Investigation:
        """Normalise a raw metadata dictionary into a validated Investigation.

        This is the primary entry point.  It delegates to field-specific
        helpers and constructs a fully validated Pydantic model.

        Args:
            raw: Dictionary of raw investigation metadata as produced by
                 an ingestion scraper.

        Returns:
            A validated :class:`Investigation` instance.

        Raises:
            pydantic.ValidationError: If required fields are missing or
                invalid after normalisation.
        """
        investigators = [
            self.normalize_investigator(inv_raw)
            for inv_raw in (raw.get("investigators") or [])
            if self.clean_text(inv_raw.get("name", ""))
        ]

        versions = [
            self.normalize_version(v_raw)
            for v_raw in (raw.get("versions") or [])
        ]

        categories = self._resolve_categories(raw.get("categories") or [])
        predicted = self._resolve_categories(raw.get("predicted_categories") or [])

        return Investigation(
            id=self.clean_text(raw.get("id", "")),
            title=self.clean_text(raw.get("title", "")),
            full_name=self.clean_text(raw.get("full_name", raw.get("title", ""))),
            description=self.clean_multiline(raw.get("description", "")),
            objectives=[
                self.clean_text(o)
                for o in (raw.get("objectives") or [])
                if self.clean_text(o)
            ],
            categories=categories,
            predicted_categories=predicted or None,
            category_confidence=raw.get("category_confidence"),
            status=self._resolve_enum(
                InvestigationStatus, raw.get("status", ""), InvestigationStatus.ACTIVE
            ),
            platform=self._resolve_enum(
                SpacecraftPlatform, raw.get("platform", ""), SpacecraftPlatform.OTHER
            ),
            facility=self.clean_text(raw.get("facility")) or None,
            start_date=self.parse_date(raw.get("start_date")),
            end_date=self.parse_date(raw.get("end_date")),
            investigators=investigators,
            source_url=self.validate_url(raw.get("source_url")) or "https://psi.nasa.gov",
            psi_url=self.validate_url(raw.get("psi_url")),
            versions=versions,
            active_version=raw.get("active_version"),
            keywords=self.normalize_keywords(raw.get("keywords")),
            related_investigations=[
                self.clean_text(r)
                for r in (raw.get("related_investigations") or [])
                if self.clean_text(r)
            ],
            applications=[
                self.clean_text(a)
                for a in (raw.get("applications") or [])
                if self.clean_text(a)
            ],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    # -----------------------------------------------------------------------
    # Merge
    # -----------------------------------------------------------------------

    def merge(
        self,
        existing: Investigation,
        incoming: Investigation,
    ) -> Investigation:
        """Merge an incoming investigation record into an existing one.

        Fields in *incoming* take precedence unless they are empty / ``None``,
        in which case the *existing* value is preserved.  List fields
        (keywords, objectives, applications, related_investigations) are
        unioned.  Versions from *incoming* that are not already present
        are appended.

        Args:
            existing: The current catalogue record.
            incoming: The newly ingested record.

        Returns:
            A merged :class:`Investigation`.
        """
        merged_keywords = list(
            dict.fromkeys(existing.keywords + incoming.keywords)
        )
        merged_objectives = list(
            dict.fromkeys(existing.objectives + incoming.objectives)
        )
        merged_applications = list(
            dict.fromkeys(existing.applications + incoming.applications)
        )
        merged_related = list(
            dict.fromkeys(
                existing.related_investigations + incoming.related_investigations
            )
        )

        # Merge investigators by name (deduplicate).
        inv_by_name: dict[str, Investigator] = {}
        for inv in existing.investigators + incoming.investigators:
            inv_by_name[inv.name.lower()] = inv
        merged_investigators = list(inv_by_name.values())

        # Merge versions by version_number.
        ver_by_num: dict[int, InvestigationVersion] = {}
        for v in existing.versions:
            ver_by_num[v.version_number] = v
        for v in incoming.versions:
            ver_by_num[v.version_number] = v
        merged_versions = sorted(ver_by_num.values(), key=lambda v: v.version_number)

        # Prefer non-empty values from incoming.
        def _prefer(incoming_val: Any, existing_val: Any) -> Any:
            if incoming_val is None or incoming_val == "" or incoming_val == []:
                return existing_val
            return incoming_val

        merged_categories = list(
            dict.fromkeys(existing.categories + incoming.categories)
        )

        return Investigation(
            id=existing.id,
            title=_prefer(incoming.title, existing.title),
            full_name=_prefer(incoming.full_name, existing.full_name),
            description=_prefer(incoming.description, existing.description),
            objectives=merged_objectives,
            categories=merged_categories,
            predicted_categories=_prefer(
                incoming.predicted_categories, existing.predicted_categories
            ),
            category_confidence=_prefer(
                incoming.category_confidence, existing.category_confidence
            ),
            status=_prefer(incoming.status, existing.status),
            platform=_prefer(incoming.platform, existing.platform),
            facility=_prefer(incoming.facility, existing.facility),
            start_date=_prefer(incoming.start_date, existing.start_date),
            end_date=_prefer(incoming.end_date, existing.end_date),
            investigators=merged_investigators,
            source_url=_prefer(incoming.source_url, existing.source_url),
            psi_url=_prefer(incoming.psi_url, existing.psi_url),
            versions=merged_versions,
            active_version=(
                max((v.version_number for v in merged_versions), default=None)
            ),
            keywords=merged_keywords,
            related_investigations=merged_related,
            applications=merged_applications,
            created_at=existing.created_at,
            updated_at=datetime.now(timezone.utc),
            has_embeddings=existing.has_embeddings or incoming.has_embeddings,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _resolve_categories(
        self, raw_categories: list[str | ResearchCategory],
    ) -> list[ResearchCategory]:
        """Resolve a mixed list of strings/enums to ``ResearchCategory``."""
        result: list[ResearchCategory] = []
        seen: set[ResearchCategory] = set()
        for item in raw_categories:
            if isinstance(item, ResearchCategory):
                cat = item
            else:
                cat = ResearchCategory.from_string(str(item))
            if cat not in seen:
                seen.add(cat)
                result.append(cat)
        return result

    @staticmethod
    def _resolve_enum(enum_cls: type, value: str | Any, default: Any) -> Any:
        """Resolve a string to an enum member, falling back to *default*.

        Args:
            enum_cls: The target enum class.
            value: String value to resolve.
            default: Fallback if resolution fails.

        Returns:
            An enum member.
        """
        if isinstance(value, enum_cls):
            return value
        if not value:
            return default
        normalised = str(value).strip().lower().replace(" ", "_").replace("-", "_")
        for member in enum_cls:
            if normalised in (member.name.lower(), member.value.lower()):
                return member
        return default
