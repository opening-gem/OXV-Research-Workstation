"""Loading a file, optionally running an adapter, and guessing key columns."""
import io
import re
import pandas as pd
from pathlib import Path
try:
    from flame_project.config import KEYWORDS
    from flame_project.adapters import ADAPTERS, auto_adapter
except ImportError:
    from config import KEYWORDS
    from adapters import ADAPTERS, auto_adapter


def _read_bytes(src):
    if isinstance(src, (bytes, bytearray)):
        return bytes(src)
    if hasattr(src, "getvalue"):          # Streamlit UploadedFile / BytesIO
        return src.getvalue()
    if hasattr(src, "read"):
        return src.read()
    return Path(src).read_bytes()


def _decode(raw):
    """Try common encodings; Excel-exported CSVs are often cp1252, not UTF-8."""
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _delimiter(text):
    """Pick the most frequent of , tab ; | in the header line (the file's own
    ';' inside column names like 'pressure; mmHg' cannot outvote real commas)."""
    head = text.splitlines()[0] if text else ""
    return max([",", "\t", ";", "|"], key=head.count)


def load_data(path_or_buffer, name="", fmt="Auto"):
    """fmt: 'Auto' | 'Generic (already tidy)' | any key of ADAPTERS."""
    name = name or str(path_or_buffer)
    if name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path_or_buffer)
    else:
        text = _decode(_read_bytes(path_or_buffer))
        df = pd.read_csv(io.StringIO(text), sep=_delimiter(text))
    df.columns = [str(c).strip() for c in df.columns]
    if fmt == "Auto":
        fmt = auto_adapter(df) or "Generic (already tidy)"
    if fmt in ADAPTERS:
        df = ADAPTERS[fmt](df)
    df.attrs["format"] = fmt
    return df


def _score(col, kws):
    name = re.sub(r"[^a-z0-9]", "_", str(col).lower())
    return max((len(k) / (1 + name.find(k)) for k in kws if k in name), default=0)


def detect_columns(df, oxygen=None, airflow=None, outcome=None):
    """Return dict(oxygen, airflow, outcome). Explicit args > adapter hints > keyword guess."""
    hint = df.attrs.get("suggested", {})
    found = {"oxygen": oxygen or hint.get("oxygen"), "airflow": airflow or hint.get("airflow"),
             "outcome": outcome or hint.get("outcome")}
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    used = {v for v in found.values() if v}

    for role in ("oxygen", "airflow"):
        if found[role]:
            continue
        cands = sorted([c for c in numeric if c not in used],
                       key=lambda c: -_score(c, KEYWORDS[role]))
        if cands:
            found[role] = cands[0]
            used.add(cands[0])

    if not found["outcome"]:
        rest = [c for c in df.columns if c not in used]
        rest.sort(key=lambda c: (-_score(c, KEYWORDS["outcome"]), df[c].nunique()))
        found["outcome"] = rest[0] if rest else None
    return found


def default_labels(df, outcome_col):
    """(sustained_labels, extinct_labels) from adapter hints, else (None, None)."""
    if df.attrs.get("suggested", {}).get("outcome") == outcome_col and "sustained_labels" in df.attrs:
        return df.attrs["sustained_labels"], df.attrs["extinct_labels"]
    return None, None
