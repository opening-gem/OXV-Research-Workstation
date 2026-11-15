"""Central settings. Change defaults here; nothing else is dataset-specific."""
DEFAULTS = {
    "n_bins": 8,          # bins per axis (continuous variables)
    "min_samples": 3,     # hide cells/bins with fewer rows
    "output_dir": "outputs",
}

# keywords used to auto-detect columns by name
KEYWORDS = {
    "oxygen": ["oxygen", "o2", "oxid", "o_2"],
    "airflow": ["airflow", "air_flow", "flow", "velocity", "speed", "wind", "air"],
    "outcome": ["outcome", "flame", "burn", "extinct", "sustain", "result",
                "status", "label", "state", "ignit"],
}

# prefixes used to auto-map outcome text -> 1 (sustained) / 0 (extinct)
POS_PREFIX = ("sustain", "burn", "flame", "ignit", "stable", "true", "yes",
              "success", "alive", "continu", "persist")
NEG_PREFIX = ("exting", "extinct", "quench", "blow", "fail", "false", "no", "non",
              "not", "un", "out", "stop", "dead", "off", "die")
