"""Berechnet aus dem Wide-Format (Var_1..Var_n je Tensorblock) Mittelwert und
Konfidenzintervall pro Zeile (Kombination, Iteration) - Grundlage fuer
Konvergenzgeschwindigkeits-Plots je Kombination.

Liest die von pivot_simple_regret.py erzeugte Wide-Format-Datei (z.B.
data/ergebnisse_wide.xlsx, eine Zeile je Kombination+Iteration, Tensorbloecke
nebeneinander in Var_1..Var_n) und ersetzt die Var_1..Var_n-Spalten durch
mean/std/ci_lower/ci_upper - berechnet je Zeile ueber die Werte der
Tensorbloecke hinweg (die statistischen Wiederholungen, N_SAMPLE_STAT in
config.py). Ergebnis ist wieder eine Wide-Format-Datei: eine Zeile je
Kombination+Iteration, jetzt mit den Konfidenzintervall-Spalten statt Var_n.

    python Post_Processing/compute_confidence_interval.py --in data/ergebnisse_wide.xlsx --out data/ergebnisse_konfidenzintervall.xlsx
    python Post_Processing/compute_confidence_interval.py --in data/ergebnisse_wide.xlsx --out data/ergebnisse_konfidenzintervall.xlsx --konfidenz 0.99
    (aus der ROBO_Benchmark-Wurzel)
"""

import argparse
import re

import numpy as np
import pandas as pd
from scipy import stats

ID_COLS = [
    "combination_id", "funktion", "dimension", "surrogate_model",
    "acquisition_model", "sample_size_initial", "max_iteration", "noise_config",
    "iteration",
]

VAR_COL_PATTERN = re.compile(r"^Var_\d+$")


def compute_row_ci(values, konfidenz):
    """Mittelwert + (Konfidenz*100)%-Konfidenzintervall fuer eine Zeile (n Werte,
    je einer pro Tensorblock).

    Nutzt die t-Verteilung statt der Normalverteilung, da n (Anzahl Tensorbloecke,
    N_SAMPLE_STAT in config.py) typischerweise klein ist - mit wenigen Freiheitsgraden
    ist das Konfidenzintervall der Normalverteilung zu optimistisch (zu schmal).

    Gibt (mean, std, ci_lower, ci_upper, n) zurueck. Bei n<2 ist keine Streuung
    schaetzbar: std/Intervallbreite werden dann als NaN markiert statt eines
    irrefuehrenden Punktintervalls.
    """
    values = values[~np.isnan(values)]
    n = len(values)

    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0

    mean = float(np.mean(values))
    if n < 2:
        return mean, np.nan, np.nan, np.nan, n

    std = float(np.std(values, ddof=1))
    sem = std / np.sqrt(n)
    alpha = 1.0 - konfidenz
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
    halfwidth = t_crit * sem

    return mean, std, mean - halfwidth, mean + halfwidth, n


def main(in_path, out_path, konfidenz):
    df = pd.read_excel(in_path) if in_path.endswith((".xlsx", ".xls")) else pd.read_csv(in_path)

    var_cols = [c for c in df.columns if VAR_COL_PATTERN.match(c)]
    if not var_cols:
        raise SystemExit(f"Keine Var_n-Spalten in {in_path} gefunden - falsches Format? "
                          f"Erwartet wird die Ausgabe von pivot_simple_regret.py.")
    id_cols = [c for c in ID_COLS if c in df.columns]

    print(f"Zeilen: {len(df)} | Tensorbloecke (Var-Spalten): {len(var_cols)} | "
          f"Konfidenzniveau: {konfidenz:.0%}")

    values_matrix = df[var_cols].to_numpy(dtype=float)
    results = [compute_row_ci(row, konfidenz) for row in values_matrix]
    means, stds, ci_lowers, ci_uppers, ns = zip(*results)

    df_out = df[id_cols].copy()
    df_out["n_wiederholungen"] = ns
    df_out["mean"] = means
    df_out["std"] = stds
    df_out["ci_lower"] = ci_lowers
    df_out["ci_upper"] = ci_uppers
    df_out["ci_halfwidth"] = [
        (u - l) / 2 if pd.notna(u) and pd.notna(l) else np.nan
        for l, u in zip(ci_lowers, ci_uppers)
    ]

    n_incomplete = sum(1 for n in ns if n < len(var_cols))
    if n_incomplete:
        print(f"Hinweis: {n_incomplete} Zeile(n) hatten weniger als {len(var_cols)} "
              f"gueltige Werte (NaN/fehlgeschlagene Tasks wurden ausgeschlossen).")

    if out_path.endswith(".xlsx"):
        df_out.to_excel(out_path, index=False)
    else:
        df_out.to_csv(out_path, index=False)
    print(f"Geschrieben nach: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True,
                         help="Wide-Format-Datei von pivot_simple_regret.py (Var_1..Var_n)")
    parser.add_argument("--out", default="data/ergebnisse_konfidenzintervall.xlsx")
    parser.add_argument("--konfidenz", type=float, default=0.95,
                         help="Konfidenzniveau, z.B. 0.95 fuer ein 95%%-Konfidenzintervall")
    args = parser.parse_args()
    main(args.in_path, args.out, args.konfidenz)
