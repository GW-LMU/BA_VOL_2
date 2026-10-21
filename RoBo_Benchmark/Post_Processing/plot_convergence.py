"""Erstellt einen Konvergenzgeschwindigkeits-Plot (Mittelwert + Konfidenzband
ueber die Iteration) aus der Ausgabe von compute_confidence_interval.py.

Einfache Filterlogik: ueber --funktion/--dimension/--surrogate-model/
--acquisition-model wird auf eine oder mehrere Kombinationen eingegrenzt.
Bleiben nach dem Filtern mehrere Kombinationen uebrig (z.B. verschiedene
surrogate_model fuer dieselbe Funktion/Dimension), wird pro Kombination eine
eigene Linie + Konfidenzband gezeichnet - so laesst sich die
Konvergenzgeschwindigkeit mehrerer Varianten direkt vergleichen.

    # eine einzelne Kombination (aus der ROBO_Benchmark-Wurzel)
    python Post_Processing/plot_convergence.py --in data/ergebnisse_konfidenzintervall.xlsx --funktion 1 --dimension 2 --surrogate-model GausianProzess --acquisition-model EI --out plots/f1_d2_gp.png

    # mehrere Varianten fuer dieselbe Funktion/Dimension zum Vergleich
    python Post_Processing/plot_convergence.py --in data/ergebnisse_konfidenzintervall.xlsx --funktion 1 --dimension 2 --out plots/f1_d2_vergleich.png

    # ein eigener Plot pro in den Daten vorhandener (Funktion, Dimension) statt
    # nur einer einzelnen --dimension - Dateien landen als f{funktion}_d{dimension}.png
    # in --out-dir (Default: plots/); --dimension darf dabei nicht gesetzt sein.
    python Post_Processing/plot_convergence.py --in data/ergebnisse_konfidenzintervall.xlsx --per-dimension --out-dir plots

    # Konfidenzband als duenne Errorbar-Striche pro Iteration statt als
    # durchgezogene Flaeche (Ax/BoTorch-Tutorial-Optik)
    python Post_Processing/plot_convergence.py --in data/ergebnisse_konfidenzintervall.xlsx --per-dimension --band-style errorbar --out-dir plots/errorbar
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # funktioniert ohne Display (z.B. auf der LRZ-Instanz/SSH)
import matplotlib.pyplot as plt
import pandas as pd

GROUP_COLS = ["funktion", "dimension", "surrogate_model", "acquisition_model"]


def load_data(in_path):
    if in_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(in_path)
    return pd.read_csv(in_path)


def filter_data(df, funktion=None, dimension=None, surrogate_model=None, acquisition_model=None):
    """Einfache Filterlogik: jeder angegebene Parameter grenzt df weiter ein,
    nicht angegebene Parameter (None) bleiben ungefiltert."""
    mask = pd.Series(True, index=df.index)
    if funktion is not None:
        mask &= df["funktion"] == funktion
    if dimension is not None:
        mask &= df["dimension"] == dimension
    if surrogate_model is not None:
        mask &= df["surrogate_model"] == surrogate_model
    if acquisition_model is not None:
        mask &= df["acquisition_model"] == acquisition_model
    return df[mask].copy()


def plot_convergence(df, out_path=None, title=None, ylabel="simple_regret", band_style="fill"):
    """Zeichnet fuer jede in df verbleibende (funktion, dimension, surrogate_model,
    acquisition_model)-Kombination eine Linie (mean) + Konfidenzband
    (ci_lower/ci_upper) ueber der Iteration.

    band_style:
      "fill"      - durchgezogene schattierte Flaeche (Standard, fill_between).
      "errorbar"  - Konfidenzband als duenner vertikaler Strich PRO Iteration
                    (ci_lower..ci_upper), ohne Kappen - Ax/BoTorch-Tutorial-Optik.
    """
    if df.empty:
        raise SystemExit(
            "Keine Zeilen nach dem Filtern uebrig - Filterwerte pruefen "
            "(funktion/dimension/surrogate_model/acquisition_model)."
        )
    if band_style not in ("fill", "errorbar"):
        raise ValueError(f"Unbekannter band_style: {band_style} (erwartet: 'fill' oder 'errorbar')")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.get_cmap("tab10").colors

    for i, (key, gdf) in enumerate(df.groupby(GROUP_COLS, sort=True)):
        gdf = gdf.sort_values("iteration")
        funktion, dimension, surrogate_model, acquisition_model = key
        label = f"{surrogate_model} / {acquisition_model} (f{funktion}, d{dimension})"
        color = colors[i % len(colors)]

        # Konfidenzband nur zeichnen, wo Grenzen vorhanden sind - bei n<2
        # Tensorbloecken sind ci_lower/ci_upper NaN (siehe compute_confidence_interval.py)
        has_ci = gdf["ci_lower"].notna() & gdf["ci_upper"].notna()
        if has_ci.any():
            if band_style == "fill":
                ax.fill_between(
                    gdf.loc[has_ci, "iteration"],
                    gdf.loc[has_ci, "ci_lower"],
                    gdf.loc[has_ci, "ci_upper"],
                    color=color, alpha=0.18, linewidth=0,
                )
            else:  # "errorbar": ein vertikaler Strich je Iteration statt einer Flaeche,
                # mit einem Punkt an jedem Ende des Strichs (ci_lower/ci_upper)
                mean_ci = gdf.loc[has_ci, "mean"]
                iters_ci = gdf.loc[has_ci, "iteration"]
                lower_err = mean_ci - gdf.loc[has_ci, "ci_lower"]
                upper_err = gdf.loc[has_ci, "ci_upper"] - mean_ci
                ax.errorbar(
                    iters_ci, mean_ci,
                    yerr=[lower_err, upper_err],
                    fmt="none", ecolor=color, elinewidth=7.5, alpha=0.5,
                    capsize=0, zorder=1,
                )
                ax.scatter(iters_ci, gdf.loc[has_ci, "ci_lower"], color=color, s=14, alpha=0.5, zorder=1)
                ax.scatter(iters_ci, gdf.loc[has_ci, "ci_upper"], color=color, s=14, alpha=0.5, zorder=1)

        ax.plot(gdf["iteration"], gdf["mean"], color=color, label=label, linewidth=3.6, zorder=2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(f"{ylabel} (Mittelwert ± Konfidenzintervall)")
    ax.set_title(title or "Konvergenzgeschwindigkeit")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Plot gespeichert: {out_path}")
    else:
        fig.savefig("convergence_plot.png", dpi=150)
        print("Kein --out angegeben, gespeichert als: convergence_plot.png")

    return fig, ax


def plot_convergence_per_dimension(df, out_dir="plots", title=None, ylabel="simple_regret", band_style="fill"):
    """Wie plot_convergence(), aber ein eigener Plot (+ eigene PNG-Datei) pro
    in df vorhandener (funktion, dimension)-Kombination statt einem einzigen
    Plot ueber alle Zeilen. Dateiname jeweils f{funktion}_d{dimension}.png in
    out_dir. Innerhalb einer (funktion, dimension)-Kombination wird weiterhin
    wie gehabt pro surrogate_model/acquisition_model eine Linie gezeichnet."""
    if df.empty:
        raise SystemExit(
            "Keine Zeilen nach dem Filtern uebrig - Filterwerte pruefen "
            "(funktion/surrogate_model/acquisition_model)."
        )

    pairs = df[["funktion", "dimension"]].drop_duplicates().sort_values(["funktion", "dimension"])
    for _, (funktion, dimension) in pairs.iterrows():
        gdf = df[(df["funktion"] == funktion) & (df["dimension"] == dimension)]
        out_path = Path(out_dir) / f"f{funktion}_d{dimension}.png"
        plot_title = title or f"Konvergenzgeschwindigkeit (Funktion {funktion}, Dimension {dimension})"

        fig, ax = plot_convergence(gdf, out_path=str(out_path), title=plot_title, ylabel=ylabel, band_style=band_style)
        plt.close(fig)  # Speicher freigeben, bevor die naechste Dimension gezeichnet wird


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True,
                         help="Ausgabe von compute_confidence_interval.py (mean/ci_lower/ci_upper je Zeile)")
    parser.add_argument("--funktion", type=int, default=None)
    parser.add_argument("--dimension", type=int, default=None)
    parser.add_argument("--surrogate-model", default=None)
    parser.add_argument("--acquisition-model", default=None)
    parser.add_argument("--out", default=None,
                         help="Pfad fuer die PNG-Datei, z.B. plots/f1_d2.png (Default: convergence_plot.png)")
    parser.add_argument("--per-dimension", action="store_true",
                         help="Statt eines einzigen Plots einen eigenen Plot pro in den Daten "
                              "vorhandener Dimension erzeugen (Dateien f{funktion}_d{dimension}.png "
                              "in --out-dir). Schliesst sich mit --dimension/--out aus.")
    parser.add_argument("--out-dir", default="plots",
                         help="Zielverzeichnis fuer --per-dimension (Default: plots).")
    parser.add_argument("--title", default=None)
    parser.add_argument("--ylabel", default="simple_regret",
                         help="Beschriftung der y-Achse / Name der geplotteten Metrik "
                              "(Standard: simple_regret; z.B. 'Mean Best Target Value' "
                              "wenn --in aus pivot_simple_regret.py --value-col best_so_far stammt).")
    parser.add_argument("--band-style", default="fill", choices=["fill", "errorbar"],
                         help="'fill' (Standard) = durchgezogene Konfidenzflaeche; "
                              "'errorbar' = duenner vertikaler Strich pro Iteration statt Flaeche.")
    args = parser.parse_args()

    if args.per_dimension and args.dimension is not None:
        raise SystemExit("--dimension und --per-dimension schliessen sich aus: bei --per-dimension "
                          "wird automatisch je vorhandener Dimension ein eigener Plot erzeugt.")

    df = load_data(args.in_path)
    df_filtered = filter_data(
        df, funktion=args.funktion, dimension=args.dimension,
        surrogate_model=args.surrogate_model, acquisition_model=args.acquisition_model,
    )
    n_groups = df_filtered.groupby(GROUP_COLS).ngroups if not df_filtered.empty else 0
    print(f"{len(df_filtered)} von {len(df)} Zeilen nach Filter, {n_groups} Kombination(en)")

    if args.per_dimension:
        plot_convergence_per_dimension(df_filtered, out_dir=args.out_dir, title=args.title,
                                        ylabel=args.ylabel, band_style=args.band_style)
    else:
        plot_convergence(df_filtered, out_path=args.out, title=args.title,
                          ylabel=args.ylabel, band_style=args.band_style)


if __name__ == "__main__":
    main()
