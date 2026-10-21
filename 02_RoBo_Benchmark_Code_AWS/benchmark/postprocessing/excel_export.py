# postprocessing/excel_export.py
#
# Erstellt aus der Ergebnis-CSV des experiment_runner eine Excel-Datei.
# Grundlage ist immer die KOMPLETTE CSV (auch Ergebnisse frueherer, fortgesetzter Laeufe).
#
# Blaetter:
#   Daten        - alle Zeilen, ohne die langen JSON-Spalten (Modell, Startdaten)
#   Verlauf      - Mittelwert/Std ueber die Wiederholungen je Iteration
#                  (z. B. Mean Optimization Path nach Rodemann & Augustin)
#   Endwerte     - wie Verlauf, aber nur die letzte Iteration jedes Experiments
#   Fehler       - Experimente mit status == "error" (nur falls vorhanden)

from pathlib import Path

import pandas as pd


# Lange JSON-Spalten - bleiben in der CSV, gehoeren aber nicht in Excel
JSON_COLUMNS = [
    "mu_train", "sigma_train", "hyperparameters",
    "start_X", "start_f_obs", "start_f_true",
]

# Ein Experiment-Setup ohne Wiederholung -> ueber diese Spalten wird gemittelt
GROUP_COLUMNS = [
    "algorithm", "surrogate_model", "acquisition_func",
    "function", "dimension", "instance", "scenario", "intensity",
]

VALUE_COLUMNS = [
    "next_f_true", "best_observed_so_far", "best_true_so_far",
    "simple_regret", "instantaneous_regret", "cumulative_regret",
    "M1", "M2", "M3", "M4", "M5", "M6", "iter_time_s",
]

EXCEL_MAX_ROWS = 1_048_575  # ohne Kopfzeile


def _aggregate(df, group_columns):
    values = [c for c in VALUE_COLUMNS if c in df.columns]
    grouped = df.groupby(group_columns, dropna=False)
    summary = grouped[values].agg(["mean", "std"])
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary.insert(0, "n_repetitions", grouped["repetition"].nunique())
    return summary.reset_index()


def export_excel(csv_path, xlsx_path=None):
    csv_path = Path(csv_path)
    xlsx_path = Path(xlsx_path) if xlsx_path is not None else csv_path.with_suffix(".xlsx")

    header = pd.read_csv(csv_path, nrows=0).columns
    df = pd.read_csv(csv_path, usecols=[c for c in header if c not in JSON_COLUMNS])

    ok = df[df["status"] == "ok"]
    # Fehler, die bei einem spaeteren (fortgesetzten) Lauf erfolgreich nachgerechnet wurden, weglassen
    errors = df[(df["status"] != "ok") & ~df["task_id"].isin(ok["task_id"])]

    group_columns = [c for c in GROUP_COLUMNS if c in df.columns]
    progress = _aggregate(ok, group_columns + ["iteration"])

    last_iteration = ok.groupby("task_id")["iteration"].transform("max")
    final = _aggregate(ok[ok["iteration"] == last_iteration], group_columns)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        if len(df) <= EXCEL_MAX_ROWS:
            df.to_excel(writer, sheet_name="Daten", index=False)
        else:
            print(f"Blatt 'Daten' uebersprungen: {len(df)} Zeilen > Excel-Limit "
                  f"({EXCEL_MAX_ROWS}). Rohdaten stehen in {csv_path}.")
        progress.to_excel(writer, sheet_name="Verlauf", index=False)
        final.to_excel(writer, sheet_name="Endwerte", index=False)
        if not errors.empty:
            errors.to_excel(writer, sheet_name="Fehler", index=False)

    return xlsx_path
