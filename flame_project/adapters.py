"""Dataset adapters: turn a messy raw file into a TIDY table the rest of the
project understands. Add a new function + register it in ADAPTERS to support
another dataset format. The adapter also returns hints via df.attrs:
    suggested         -> default oxygen/airflow/outcome columns
    sustained_labels  -> outcome labels meaning sustained burning
    extinct_labels    -> outcome labels meaning NOT sustained
    notes             -> caveats shown in the UI
"""
import re
import numpy as np
import pandas as pd

NUM = re.compile(r"\d*\.\d+|\d+")
MAX_CM_S = 40          # values above this in 'Air display' are fan counts, not cm/s


def _nums(s):
    return [float(x) for x in NUM.findall(s)]


def _clean_cols(df):
    df = df.copy()
    df.columns = [" ".join(str(c).split()) for c in df.columns]
    return df


# ------------------------------------------------------------------ BASS-II --
def _airflow_series(fan, air):
    """cm/s values listed in the free-text 'Air display' (fallback: '5 cm/s' in Fan text)."""
    s = "" if pd.isna(air) else str(air).lower()
    outside = re.sub(r"\([^)]*\)", " ", s)
    for chunk in [outside] + re.findall(r"\(([^)]*)\)", s):
        vals = [v for v in _nums(chunk) if v <= MAX_CM_S]
        if vals:
            return vals
    m = re.search(r"(\d*\.?\d+)\s*cm/s", "" if pd.isna(fan) else str(fan).lower())
    return [float(m.group(1))] if m else []


def _outcome(fan, air):
    t = f"{fan} {air}".lower()
    if re.search(r"\bno ign\w*", t):
        return "No ignition"
    if re.search(r"\bblow\s?off\b", t):
        return "Blowoff"
    if re.search(r"\bext\b|\bexting\w*|\bout\b|would not stay lit|\boff\b", t):
        return "Extinguished"
    return "Sustained"


def _fuel_type(s):
    s = str(s).lower()
    for key, name in [("sibal", "SIBAL"), ("nomex", "Nomex"), ("candle", "Wax candle"),
                      ("spher", "PMMA sphere"), ("rod", "PMMA rod")]:
        if key in s:
            return name
    return "PMMA film/slab" if re.search(r"micron|mm|film|slab", s) else "Other"


def _thickness_um(s):
    s = str(s).lower()
    m = re.search(r"(\d*\.?\d+)\s*(micron|µm|um)", s)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d*\.?\d+)\s*mm", s)
    return float(m.group(1)) * 1000 if m else np.nan


def bass2(raw):
    df = _clean_cols(raw)
    out = pd.DataFrame({"test_id": df["Test #"], "pi": df["PI"]})
    out["fuel_type"] = df["Fuel Sample Material"].map(_fuel_type)
    out["thickness_um"] = df["Fuel Sample Material"].map(_thickness_um)
    out["flow_restrictor"] = df["Flow restrictor"].astype(str)
    out["o2_initial"] = pd.to_numeric(df["Calibrated initial O2 % by vol"], errors="coerce")
    out["o2_final"] = pd.to_numeric(df["Calibrated final O2 % by vol"], errors="coerce")
    out["outcome"] = [_outcome(f, a) for f, a in zip(df["Fan display"], df["Air display"])]

    series = [_airflow_series(f, a) for f, a in zip(df["Fan display"], df["Air display"])]
    first = lambda v: v[0] if v else np.nan
    out["airflow_start"] = [first(v) for v in series]
    out["airflow_min"] = [min(v) if v else np.nan for v in series]
    out["airflow_max"] = [max(v) if v else np.nan for v in series]
    # "critical" flow: extinction -> lowest flow reached; blowoff -> highest flow reached;
    # otherwise the lowest flow at which burning was still recorded
    out["airflow_critical"] = np.where(out["outcome"] == "Blowoff",
                                       out["airflow_max"], out["airflow_min"])
    out.attrs.update(
        suggested={"oxygen": "o2_initial", "airflow": "airflow_critical", "outcome": "outcome"},
        sustained_labels=["Sustained"],
        extinct_labels=["Extinguished", "Blowoff", "No ignition"],
        notes=("BASS-II has no outcome column: outcomes are parsed from the free-text notes "
               "('ext', 'blowoff', 'out', 'no ign'). 'Sustained' = no extinction noted in the "
               "log. Airflow (cm/s) is parsed from 'Air display'; rows without a readable "
               "value are dropped from the O2-airflow plots."))
    return out


# ------------------------------------------------------------------- FLEX --
def flex(raw):
    """NASA FLEX (droplet combustion). No airflow exists: droplets burn in a quiescent
    atmosphere, so the 2nd axis becomes the diluent (CO2) mole fraction."""
    df = _clean_cols(raw)
    find = lambda key: next(c for c in df.columns if c.lower().startswith(key.lower()))
    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    out = pd.DataFrame({"test_id": df[find("FLEX Identifier")], "fuel": df[find("Fuel")]})
    out["pressure_mmhg"] = num(find("Ambient pressure"))
    out["o2_pct"] = num(find("O initial")) * 100
    out["n2_pct"] = num(find("N initial")) * 100
    out["co2_pct"] = num(find("CO initial")) * 100      # header says 'CO' (CO2 subscript lost)
    out["he_pct"] = num(find("He initial")) * 100
    out["droplet_d0_mm"] = num(find("Droplet initial diameter"))
    out["ext_diameter_mm"] = num(find("Visible flame extinction diameter"))
    out["burn_time_s"] = num(find("Burn time"))
    out["outcome"] = df[find("Test end")].astype(str).str.strip()
    out.attrs.update(
        suggested={"oxygen": "o2_pct", "airflow": "co2_pct", "outcome": "outcome"},
        sustained_labels=["Completion"], extinct_labels=["Extinction"],
        notes=("FLEX has NO airflow column (quiescent droplet combustion). The second axis is "
               "set to CO2 diluent % (the 'CO' column, subscript lost); pick another column in "
               "the sidebar if you prefer. Completion = droplet burned out, Extinction = flame "
               "went out first, Disruption = test disturbed (left unassigned)."))
    return out


ADAPTERS = {"BASS-II (raw)": bass2, "FLEX (raw)": flex}


def auto_adapter(df):
    cols = {" ".join(str(c).split()).lower() for c in df.columns}
    if {"fan display", "air display", "calibrated initial o2 % by vol"} <= cols:
        return "BASS-II (raw)"
    if any(c.startswith("flex test") for c in cols):
        return "FLEX (raw)"
    return None
