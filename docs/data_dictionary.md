# Flame in Freefall: Data Dictionary

This document specifies the canonical data schema models used across Flame in Freefall, implemented via Pydantic v2 in `src/catalog/schema.py`.

---

## 1. Model: `Investigation`

The primary entity representing a NASA combustion science research investigation.

| Field Name | Type | Required | Description | Example / Constraints |
| :--- | :--- | :--- | :--- | :--- |
| `id` | `str` | Yes | Unique short identifier for the investigation. | `"BASS-II"`, `"ACME-BRE"` (auto-capitalized) |
| `title` | `str` | Yes | Official display title of the experiment. | `"Burning and Suppression of Solids - II"` |
| `full_name` | `str` | Yes | Full expanded name of the project. | `"Burning and Suppression of Solids - Second Generation"` |
| `description` | `str` | Yes | Comprehensive scientific overview. | Minimum 10 characters |
| `objectives` | `list[str]` | Yes | Stated scientific and mission objectives. | `["Determine flame spread characteristics..."]` |
| `categories` | `list[ResearchCategory]` | Yes | Ground-truth research taxonomy labels. | `["spacecraft_fire_safety", "flame_spread"]` |
| `predicted_categories` | `list[ResearchCategory]` | No | Machine-learning predicted categories. | Populated by `ResearchClassifier` |
| `category_confidence` | `dict[str, float]` | No | Confidence probabilities per category. | `{"flame_spread": 0.88}` |
| `status` | `InvestigationStatus` | Yes | Project lifecycle state. | `"active"`, `"completed"`, `"planned"`, `"archived"` |
| `platform` | `SpacecraftPlatform` | Yes | Testing microgravity environment. | `"iss"`, `"ground"`, `"drop_tower"`, `"sounding_rocket"` |
| `facility` | `str` | No | Experimental chamber or rack on station. | `"Microgravity Science Glovebox (MSG)"` |
| `start_date` | `date` | No | Beginning date of flight/testing. | `2012-01-01` |
| `end_date` | `date` | No | Conclusion date of flight/testing. | `2013-12-31` (must be $\ge$ `start_date`) |
| `investigators` | `list[Investigator]` | Yes | Research team members. | List of `Investigator` models |
| `source_url` | `str` | Yes | Canonical web documentation URL. | Must start with `http://` or `https://` |
| `psi_url` | `str` | No | NASA PSI dataset repository URL. | `https://psi.nasa.gov/physci/repo/...` |
| `versions` | `list[InvestigationVersion]` | Yes | Dataset revision history tree. | Monotonically increasing numbers |
| `active_version` | `int` | Yes | Version number currently canonical. | `4` (for BASS-II) |
| `keywords` | `list[str]` | Yes | Deduplicated search tags. | `["fire safety", "microgravity"]` |
| `applications` | `list[str]` | Yes | Practical spaceflight and Earth impacts. | `["Spacecraft habitat fire safety standards"]` |
| `created_at` | `datetime` | Yes | Timestamp of catalog ingestion. | UTC ISO format |
| `updated_at` | `datetime` | Yes | Timestamp of last record update. | UTC ISO format |

---

## 2. Model: `InvestigationVersion`

Captures dataset revisions, release dates, and persistent DOIs.

| Field Name | Type | Required | Description | Example / Constraints |
| :--- | :--- | :--- | :--- | :--- |
| `version_number` | `int` | Yes | Monotonically increasing version integer. | $\ge 1$ |
| `doi` | `str` | No | Digital Object Identifier. | Matches `^10\.\d{4,9}/.*` |
| `release_date` | `date` | No | Date this revision was published by NASA. | `2013-05-15` |
| `description` | `str` | Yes | Summary of updates or new sensor data. | `"Added radiometer telemetry"` |
| `is_active` | `bool` | Yes | Indicates canonical active version. | Exactly one version per record is active |
| `retrieved_at` | `datetime` | Yes | Ingestion pipeline fetch timestamp. | UTC ISO format |

---

## 3. Model: `Investigator`

| Field Name | Type | Required | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `name` | `str` | Yes | Full investigator name. | `"Dr. Paul Ferkul"` |
| `affiliation` | `str` | No | Institutional home. | `"NASA Glenn Research Center"` |
| `role` | `str` | No | Role on experiment team. | `"PI"`, `"Co-I"`, `"Project Scientist"` |

---

## 4. Enumerations

### `ResearchCategory`
- `spacecraft_fire_safety`
- `droplet_combustion`
- `gaseous_combustion`
- `solid_fuel_combustion`
- `smoke_detection`
- `flame_spread`
- `fire_suppression`
- `material_flammability`
- `soot_formation`
- `cool_flames`
- `ignition_dynamics`
- `other`

### `SpacecraftPlatform`
- `iss` (International Space Station)
- `drop_tower` (Glenn 2.2s / 5.18s Zero-G Facility)
- `parabolic_flight` (Aircraft parabolas)
- `sounding_rocket` (Suborbital microgravity)
- `ground` (Normal 1g laboratories)
- `other`
