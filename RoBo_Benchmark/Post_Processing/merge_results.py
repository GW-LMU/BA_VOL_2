"""Fuehrt die results.csv aller Instanzen zu einer Gesamtdatei zusammen.

Nachdem alle 4 Ergebnisordner (bzw. deren results.csv) zurueckkopiert wurden:

    python Post_Processing/merge_results.py --pattern "data/instances/instance_*/results/results.csv"
    (aus der ROBO_Benchmark-Wurzel)
"""

import argparse
import glob

import pandas as pd

from derived_metrics import add_derived_metrics


def main(pattern, out_path):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"Keine Dateien gefunden fuer Pattern: {pattern}")

    dfs = [pd.read_csv(f) for f in files]
    df_all = add_derived_metrics(pd.concat(dfs, ignore_index=True))

    n_error = (df_all["status"] == "error").sum()
    print(f"Zeilen gesamt: {len(df_all)} | Fehlerhafte Tasks: {n_error} | Dateien: {len(files)}")

    if out_path.endswith(".xlsx"):
        df_all.to_excel(out_path, index=False)
    else:
        df_all.to_csv(out_path, index=False)
    print(f"Geschrieben nach: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--out", default="data/ergebnisse_gesamt.xlsx")
    args = parser.parse_args()
    main(args.pattern, args.out)
