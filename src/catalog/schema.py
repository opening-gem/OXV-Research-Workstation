"""
Pydantic v2 data models for NASA microgravity combustion investigations.

This module defines the canonical schema for the Flame in Freefall catalogue.
Every investigation ingested from NASA sources is validated against these models
before being persisted.

Typical usage::

    from src.catalog.schema import Investigation, ResearchCategory

    inv = Investigation(
        id="BASS-II",
        title="BASS-II",
        full_name="Burning And Suppression of Solids — II",
        description="Studies flame spread over solid fuels in microgravity.",
        objectives=["Measure flame spread rates"],
        categories=[ResearchCategory.FLAME_SPREAD],
        status=InvestigationStatus.COMPLETED,
        platform=SpacecraftPlatform.ISS,
        source_url="https://psi.nasa.gov/...",
        investigators=[],
        versions=[],
        keywords=["flame spread", "solid fuel"],
        related_investigations=[],
        applications=["Spacecraft fire safety"],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
"""

from __future__ import annotations

import re
import textwrap
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Self

from src.catalog.compat_pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ResearchCategory(str, Enum):
    """NASA combustion research categories."""

    SPACECRAFT_FIRE_SAFETY = "spacecraft_fire_safety"
    DROPLET_COMBUSTION = "droplet_combustion"
    GASEOUS_COMBUSTION = "gaseous_combustion"
    SOLID_FUEL_COMBUSTION = "solid_fuel_combustion"
    SMOKE_DETECTION = "smoke_detection"
    FLAME_SPREAD = "flame_spread"
    FIRE_SUPPRESSION = "fire_suppression"
    MATERIAL_FLAMMABILITY = "material_flammability"
    SOOT_FORMATION = "soot_formation"
    COOL_FLAMES = "cool_flames"
    IGNITION_DYNAMICS = "ignition_dynamics"
    OTHER = "other"

    @classmethod
    def from_string(cls, value: str) -> ResearchCategory:
        """Resolve a human-readable string to a category, or ``OTHER``.

        Performs case-insensitive matching against both the enum *name*
        (e.g. ``"FLAME_SPREAD"``) and the enum *value*
        (e.g. ``"flame_spread"``).
        """
        normalised = value.strip().lower().replace(" ", "_").replace("-", "_")
        for member in cls:
            if normalised in (member.name.lower(), member.value.lower()):
                return member
        return cls.OTHER


class InvestigationStatus(str, Enum):
    """Lifecycle status of an investigation."""

    ACTIVE = "active"
    COMPLETED = "completed"
    PLANNED = "planned"
    ARCHIVED = "archived"


class SpacecraftPlatform(str, Enum):
    """Platform where the experiment is (or was) conducted."""

    ISS = "iss"
    CYGNUS = "cygnus"
    GROUND = "ground"
    SOUNDING_ROCKET = "sounding_rocket"
    DROP_TOWER = "drop_tower"
    PARABOLIC_FLIGHT = "parabolic_flight"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Supporting models
# ---------------------------------------------------------------------------

class InvestigationVersion(BaseModel):
    """A specific version of an investigation's dataset.

    The PSI system publishes multiple dataset revisions.  This model captures
    each version so the catalogue can track provenance and show diffs.
    """

    version_number: int = Field(
        ..., ge=1, description="Monotonically increasing version identifier."
    )
    doi: str | None = Field(
        default=None,
        description="Digital Object Identifier for this dataset version.",
    )
    release_date: date | None = Field(
        default=None, description="Date the version was published."
    )
    description: str = Field(
        ..., min_length=1, description="Free-text description of changes in this version."
    )
    is_active: bool = Field(
        default=False,
        description="True only for the version currently considered canonical.",
    )
    retrieved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when this version was fetched by the ingestion pipeline.",
    )

    # -- validators ----------------------------------------------------------

    @field_validator("doi")
    @classmethod
    def _validate_doi(cls, v: str | None) -> str | None:
        """Ensure DOIs look roughly correct (``10.xxxx/...``)."""
        if v is not None:
            v = v.strip()
            if v and not re.match(r"^10\.\d{4,9}/", v):
                raise ValueError(
                    f"DOI '{v}' does not match the expected pattern '10.XXXX/...'."
                )
        return v or None


class Investigator(BaseModel):
    """A person associated with a combustion research investigation."""

    name: str = Field(
        ..., min_length=1, description="Full name (e.g. 'Dr. Sandra Olson')."
    )
    affiliation: str | None = Field(
        default=None, description="Institutional affiliation."
    )
    role: str | None = Field(
        default=None,
        description="Role in the investigation (e.g. 'PI', 'Co-I', 'Project Scientist').",
    )

    @field_validator("name")
    @classmethod
    def _clean_name(cls, v: str) -> str:
        """Collapse whitespace and strip surrounding blanks."""
        return " ".join(v.split())


# ---------------------------------------------------------------------------
# Central model
# ---------------------------------------------------------------------------

class Investigation(BaseModel):
    """Central model representing a NASA combustion research investigation.

    This is the single source of truth for every investigation tracked by the
    Flame in Freefall platform.  Fields are grouped into logical sections:
    identity, classification, research context, people, provenance, and
    metadata.
    """

    # -- identity ------------------------------------------------------------
    id: str = Field(
        ...,
        min_length=1,
        description="Unique short identifier (e.g. 'BASS-II', 'ACME-BRE').",
    )
    title: str = Field(..., min_length=1, description="Display title.")
    full_name: str = Field(
        ..., min_length=1, description="Full expanded name of the investigation."
    )
    description: str = Field(
        ..., min_length=1, description="Prose summary of the investigation."
    )
    objectives: list[str] = Field(
        default_factory=list, description="Explicit research objectives."
    )

    # -- classification ------------------------------------------------------
    categories: list[ResearchCategory] = Field(
        default_factory=list,
        description="Human-assigned research categories.",
    )
    predicted_categories: list[ResearchCategory] | None = Field(
        default=None,
        description="ML-predicted categories (populated by the classifier).",
    )
    category_confidence: dict[str, float] | None = Field(
        default=None,
        description="Mapping of category value → confidence score from the classifier.",
    )

    # -- research context ----------------------------------------------------
    status: InvestigationStatus = Field(
        ..., description="Current lifecycle status."
    )
    platform: SpacecraftPlatform = Field(
        ..., description="Experimental platform."
    )
    facility: str | None = Field(
        default=None,
        description="Specific facility name (e.g. 'Combustion Integrated Rack').",
    )
    start_date: date | None = Field(
        default=None, description="Date the investigation started."
    )
    end_date: date | None = Field(
        default=None, description="Date the investigation concluded."
    )

    # -- people --------------------------------------------------------------
    investigators: list[Investigator] = Field(
        default_factory=list, description="Research team members."
    )

    # -- data provenance -----------------------------------------------------
    source_url: str = Field(
        ..., min_length=1, description="URL where the investigation metadata was sourced."
    )
    psi_url: str | None = Field(
        default=None,
        description="Direct URL to the PSI dataset page, if available.",
    )
    versions: list[InvestigationVersion] = Field(
        default_factory=list, description="Dataset version history."
    )
    active_version: int | None = Field(
        default=None,
        description="Version number of the currently active dataset version.",
    )

    # -- metadata ------------------------------------------------------------
    keywords: list[str] = Field(
        default_factory=list, description="Descriptive keywords / tags."
    )
    related_investigations: list[str] = Field(
        default_factory=list,
        description="IDs of related investigations in the catalogue.",
    )
    applications: list[str] = Field(
        default_factory=list,
        description="Practical applications of this research.",
    )

    # -- timestamps ----------------------------------------------------------
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this record was first created in the catalogue.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this record was last modified.",
    )

    # -- embedding state -----------------------------------------------------
    has_embeddings: bool = Field(
        default=False,
        description="Whether vector embeddings have been generated for this investigation.",
    )

    # -----------------------------------------------------------------------
    # Validators
    # -----------------------------------------------------------------------

    @field_validator("id")
    @classmethod
    def _normalise_id(cls, v: str) -> str:
        """Strip whitespace and convert to upper-case for consistency."""
        return v.strip().upper()

    @field_validator("keywords")
    @classmethod
    def _deduplicate_keywords(cls, v: list[str]) -> list[str]:
        """Lower-case, strip, and deduplicate keywords while preserving order."""
        seen: set[str] = set()
        deduped: list[str] = []
        for kw in v:
            normalised = kw.strip().lower()
            if normalised and normalised not in seen:
                seen.add(normalised)
                deduped.append(normalised)
        return deduped

    @field_validator("objectives")
    @classmethod
    def _clean_objectives(cls, v: list[str]) -> list[str]:
        """Strip whitespace from every objective string."""
        return [obj.strip() for obj in v if obj.strip()]

    @field_validator("source_url", "psi_url")
    @classmethod
    def _validate_url(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.strip()
            if v and not re.match(r"^https?://", v):
                raise ValueError(f"URL '{v}' must start with http:// or https://.")
        return v or None

    @model_validator(mode="after")
    def _validate_dates(self) -> Self:
        """Ensure ``end_date`` is not before ``start_date``."""
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.end_date < self.start_date
        ):
            raise ValueError(
                f"end_date ({self.end_date}) must not precede "
                f"start_date ({self.start_date})."
            )
        return self

    @model_validator(mode="after")
    def _sync_active_version(self) -> Self:
        """Keep ``active_version`` and the ``is_active`` flags in sync.

        If ``active_version`` is set, exactly one version in ``versions``
        should have ``is_active=True`` with a matching ``version_number``.
        If ``active_version`` is ``None`` but versions exist, the latest
        version is promoted.
        """
        if not self.versions:
            return self

        if self.active_version is None:
            # Auto-promote the latest version.
            latest = max(self.versions, key=lambda v: v.version_number)
            self.active_version = latest.version_number

        for ver in self.versions:
            ver.is_active = ver.version_number == self.active_version

        return self

    # -----------------------------------------------------------------------
    # Public helpers
    # -----------------------------------------------------------------------

    def get_active_version(self) -> InvestigationVersion | None:
        """Return the currently active :class:`InvestigationVersion`, or ``None``.

        Returns:
            The version whose ``version_number`` matches ``self.active_version``,
            or ``None`` if no versions are recorded.
        """
        if self.active_version is None:
            return None
        for ver in self.versions:
            if ver.version_number == self.active_version:
                return ver
        return None

    def to_search_text(self) -> str:
        """Concatenate all searchable fields into a single text block.

        The returned string is suitable for TF-IDF vectorisation or
        embedding generation.  Fields are joined with newlines and
        labelled so embeddings capture field semantics.

        Returns:
            A plain-text representation of the investigation.
        """
        parts: list[str] = [
            f"Title: {self.title}",
            f"Full name: {self.full_name}",
            f"Description: {self.description}",
        ]
        if self.objectives:
            parts.append("Objectives: " + "; ".join(self.objectives))
        if self.categories:
            parts.append(
                "Categories: "
                + ", ".join(c.value.replace("_", " ") for c in self.categories)
            )
        if self.facility:
            parts.append(f"Facility: {self.facility}")
        if self.keywords:
            parts.append("Keywords: " + ", ".join(self.keywords))
        if self.applications:
            parts.append("Applications: " + ", ".join(self.applications))
        if self.investigators:
            names = [inv.name for inv in self.investigators]
            parts.append("Investigators: " + ", ".join(names))
        return "\n".join(parts)

    def add_version(self, version: InvestigationVersion) -> None:
        """Append a new version and promote it to active.

        Args:
            version: The new dataset version to add.

        Raises:
            ValueError: If a version with the same ``version_number``
                already exists.
        """
        existing_numbers = {v.version_number for v in self.versions}
        if version.version_number in existing_numbers:
            raise ValueError(
                f"Version {version.version_number} already exists for "
                f"investigation '{self.id}'."
            )
        # Deactivate all existing versions.
        for v in self.versions:
            v.is_active = False
        version.is_active = True
        self.versions.append(version)
        self.active_version = version.version_number
        self.updated_at = datetime.now(timezone.utc)

    def summary(self, width: int = 80) -> str:
        """Return a short human-readable summary of the investigation.

        Args:
            width: Maximum line width for text wrapping.

        Returns:
            A formatted multi-line string.
        """
        cats = ", ".join(c.value.replace("_", " ") for c in self.categories) or "none"
        lines = [
            f"{'=' * width}",
            f" {self.id}: {self.title}",
            f"{'=' * width}",
            f" Full name   : {self.full_name}",
            f" Status      : {self.status.value}",
            f" Platform    : {self.platform.value}",
            f" Categories  : {cats}",
            f" Versions    : {len(self.versions)}  (active: {self.active_version})",
            f" Keywords    : {', '.join(self.keywords[:8]) or 'none'}",
            "",
            textwrap.fill(self.description, width=width),
            f"{'=' * width}",
        ]
        return "\n".join(lines)
