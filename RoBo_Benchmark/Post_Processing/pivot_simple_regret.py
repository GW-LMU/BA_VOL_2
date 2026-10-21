"""Erstellt ergebnisse_wide.xlsx: eine Metrik (Standard: simple_regret) im Wide-Format.

Die normalen results.csv-Dateien sind im Long-Format (eine Zeile pro
BO-Iteration UND Tensorblock). Dieses Skript pivotiert sie so, dass jede
Zeile genau einer (Kombination, Iteration) entspricht - die Tensorbloecke
stehen dann nebeneinander in den Spalten Var_1..Var_n (statt untereinander
in eigenen Zeilen), analog zu den Var_n-Spalten in ROBO/Subprocess.py.

Ueber --value-col laesst sich statt simple_regret auch eine andere Spalte aus
results.csv pivotieren, z.B. best_so_far ("Mean Best Target Value" - der beste
bisher gefundene Zielfunktionswert, gemittelt ueber die Tensorbloecke/Wieder-
holungen; Gegenstueck zu simple_regret, siehe core_bo.py/compute_regret()).

Rauschfreie Kennzahlen (empfohlen fuer die Auswertung): regret_true (Haupt-
kennzahl), regret_true_observed, immediate_regret_true (berechnet in
derived_metrics.py) sowie robust_worstcase, robust_expected, robust_std,
stableopt_worstcase_xtilde, context_expected, context_worstcase.

    python Post_Processing/pivot_simple_regret.py --pattern "data/instances/instance_*/results/results.csv" --out data/ergebnisse_wide.xlsx
    python Post_Processing/pivot_simple_regret.py --pattern "data/instances/instance_*/results/results.csv" --value-col best_so_far --out data/ergebnisse_wide_best_so_far.xlsx
    (aus der ROBO_Benchmark-Wurzel)
"""

import argparse
import glob

import pandas as pd

from derived_metrics import add_derived_metrics

ID_COLS = [
    "combination_id", "funktion", "dimension", "surrogate_model",
    "acquisition_model", "sample_size_initial", "max_iteration", "noise_config",
    "iteration",
]


def main(pattern, out_path, value_col):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"Keine Dateien gefunden fuer Pattern: {pattern}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    n_error = (df["status"] == "error").sum()
    df = df[df["status"] == "ok"].copy()
    df = add_derived_metrics(df)

    # aeltere results.csv-Dateien (vor Einfuehrung von noise_config) haben diese
    # Spalte noch nicht - defensiv weglassen statt KeyError beim Pivotieren.
    id_cols = [c for c in ID_COLS if c in df.columns]

    df_wide = df.pivot_table(
        index=id_cols,
        columns="tensor_block",
        values=value_col,
    )
    df_wide.columns = [f"Var_{int(c) + 1}" for c in df_wide.columns]
    df_wide = df_wide.reset_index().sort_values(["combination_id", "iteration"])

    if out_path.endswith(".xlsx"):
        df_wide.to_excel(out_path, index=False)
    else:
        df_wide.to_csv(out_path, index=False)

    print(f"Metrik: {value_col}")
    print(f"Zeilen (Kombination x Iteration): {len(df_wide)} | Var-Spalten: {len(df_wide.columns) - len(ID_COLS)}")
    print(f"Ausgelassene Fehler-Tasks: {n_error}")
    print(f"Geschrieben nach: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--out", default="data/ergebnisse_wide.xlsx")
    parser.add_argument("--value-col", default="simple_regret",
                         help="Spalte aus results.csv, die pivotiert wird (Standard: simple_regret; "
                              "z.B. best_so_far fuer 'Mean Best Target Value').")
    args = parser.parse_args()
    main(args.pattern, args.out, args.value_col)
