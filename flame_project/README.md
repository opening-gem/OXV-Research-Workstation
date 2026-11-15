# Oxygen-Airflow Flame Analysis (NASA BASS-II)

    pip install -r requirements.txt
    streamlit run app.py                              # UI
    python main.py --csv data/BASS-II.csv             # saves graphs/tables to outputs/

Files: config.py | adapters.py (raw dataset -> tidy table; BASS-II parser lives here) |
data_loader.py | preprocessing.py | plots.py | analysis.py | main.py | app.py

New dataset? Either it is already tidy (Generic format, pick columns in the UI) or
add one function to adapters.py and register it in ADAPTERS.

BASS-II notes: no outcome column exists. Outcomes come from the free-text notes
(ext / blowoff / out / no ign). "Sustained" = no extinction noted. Treat it as
"no extinction recorded", not as confirmed long-term burning.
