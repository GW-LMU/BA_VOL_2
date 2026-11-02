# postprocessing/wilcoxon_auc.py
#
# Wilcoxon-Vorzeichen-Rang-Test auf der Flaeche unter der log-Regret-Kurve (AUC)
# statt auf dem Endwert. Ein Test pro Setting und Gegner fasst den ganzen
# Verlauf zusammen - keine Mehrfachtests ueber die Iterationen.
#
# AUC eines Laufs = Mittelwert von log10(m(t)) ueber alle BO-Iterationen t = 1..T
#   (bei gleichem Iterationsabstand proportional zur Flaeche unter der Kurve;
#    kleiner = im Mittel ueber den Verlauf naeher am Optimum)
#
# Paarung wie in statistics.py: Wiederholung r gegen Wiederholung r (gleiche Seeds)
#     d_r = AUC_PROBO,r - AUC_B,r
#   d < 0      ->  PROBO ueber den Verlauf besser
#   Faktor     =  10^median(d): Verhaeltnis der geometrischen Mittel des Regrets
#                 ueber den Verlauf (0.3 = PROBO im Median bei 30 % des Regrets von B)
#   r_rb       =  Rang-biseriale Korrelation (-1 = PROBO in jeder Wiederholung besser)
# Mehrfachtests: Holm-Korrektur ueber alle Tests einer Abbildung (ein Szenario).

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing import config
from postprocessing.statistics import (FIGURES_DIR, INK, MUTED, SCENARIO_LABELS,
                                       SIGNIFICANCE_NOTE, _format_factor, _holm, _log,
                                       _save_table, _stars, _wilcoxon_diff)
from postprocessing.wilcoxon_iterations import GRID, OPPONENT_COLORS, REFERENCE

RUN_KEYS = ["algorithm", "function", "dimension", "scenario", "intensity", "repetition"]
# Farbe = Richtung (wie EFFECT_CMAP in statistics.py), blass = nicht signifikant
BETTER, WORSE = "#2a78d6", "#e34948"
BETTER_NS, WORSE_NS = "#b5cff0", "#f3bcbb"


def load_auc(csv_path, metric="simple_regret"):
    """Mittelwert von log10(Metrik) ueber alle Iterationen je Lauf."""
    parts = []
    # Die CSV kann mehrere GB gross sein -> stueckweise lesen und je Lauf nur
    # Summe und Anzahl behalten (ein Lauf kann ueber zwei Stuecke verteilt sein)
    for chunk in pd.read_csv(csv_path, usecols=RUN_KEYS + ["status", metric],
                             chunksize=200_000):
        chunk = chunk[chunk["status"] == "ok"].assign(log=lambda c: _log(c[metric]))
        parts.append(chunk.groupby(RUN_KEYS)["log"].agg(["sum", "count"]))
    runs = pd.concat(parts).groupby(level=RUN_KEYS).sum().reset_index()

    # Nur vollstaendige Laeufe: unterschiedlich lange Kurven waeren nicht vergleichbar
    full = runs.groupby(["scenario", "intensity", "function", "dimension"])["count"].transform("max")
    incomplete = runs["count"] < full
    if incomplete.any():
        print(f"{incomplete.sum()} unvollstaendige Laeufe ausgeschlossen")
    runs = runs[~incomplete]
    return runs.assign(auc=runs["sum"] / runs["count"]).drop(columns="sum")


def wilcoxon_auc_table(runs, reference=REFERENCE):
    """Referenz gegen jedes andere Verfahren, je Setting (Intensitaet x Funktion x Dimension)."""
    rows = []
    for (intensity, fn, dim), sub in runs.groupby(["intensity", "function", "dimension"]):
        wide = sub.pivot_table(index="repetition", columns="algorithm", values="auc")
        if reference not in wide:
            continue
        opponents = [o for o in OPPONENT_COLORS if o in wide]
        opponents += [o for o in wide if o not in opponents and o != reference]
        for opponent in opponents:
            paired = wide[[reference, opponent]].dropna()
            if len(paired) < 2:
                continue
            rows.append({"comparison": f"{reference} vs. {opponent}", "intensity": intensity,
                         "function": fn, "dimension": dim, "A": reference, "B": opponent,
                         "n_iterations": int(sub["count"].max()),
                         **_wilcoxon_diff(paired[reference] - paired[opponent])})
    table = pd.DataFrame(rows)
    table["p_holm"] = _holm(table["p"])
    return table


def plot_auc_bars(table, title, out_path, reference=REFERENCE):
    """Zeilen = Dimension, Spalten = Funktion; je Panel ein Balken pro Gegner,
    Laenge = r_rb, sortiert (oben = Referenz am deutlichsten besser).
    Text am Balken = Median-Faktor und Holm-korrigierte Signifikanz."""
    functions = sorted(table["function"].unique())
    dims = sorted(table["dimension"].unique(), reverse=True)   # groesste Dimension oben
    n_opp = table["B"].nunique()

    fig, axes = plt.subplots(len(dims), len(functions), sharex=True,
                             figsize=(3.6 * len(functions) + 1.0,
                                      (0.38 * n_opp + 0.9) * len(dims) + 1.0),
                             squeeze=False)
    for r, dim in enumerate(dims):
        for c, fn in enumerate(functions):
            ax = axes[r, c]
            # aufsteigend nach r_rb, barh zeichnet von unten -> umdrehen fuer "bester oben"
            panel = table[(table["dimension"] == dim) & (table["function"] == fn)]
            panel = panel.sort_values("r_rb", ascending=False).reset_index(drop=True)
            significant = panel["p_holm"] < 0.05
            colors = [(BETTER if s else BETTER_NS) if v < 0 else (WORSE if s else WORSE_NS)
                      for v, s in zip(panel["r_rb"], significant)]
            y = np.arange(len(panel))
            ax.barh(y, panel["r_rb"], height=0.62, color=colors, zorder=2)
            ax.axvline(0, color="#c3c2b7", lw=0.8, zorder=1)
            for i, row in panel.iterrows():
                # Beschriftung ausserhalb der Balkenspitze, bei langen Balken innen
                inside = abs(row.r_rb) > 0.55
                x = row.r_rb + (-0.04 if row.r_rb > 0 else 0.04) * (1 if inside else -1)
                ha = ("right" if row.r_rb > 0 else "left") if inside else \
                     ("left" if row.r_rb >= 0 else "right")
                ax.text(x, i, f"×{_format_factor(row.factor)} {_stars(row.p_holm)}",
                        ha=ha, va="center", fontsize=7.5,
                        color="white" if inside and significant[i] else
                        (INK if significant[i] else MUTED),
                        fontweight="bold" if significant[i] else "normal", zorder=3)
            ax.set_yticks(y, panel["B"])
            ax.set_xlim(-1.08, 1.08)
            ax.set_ylim(-0.6, n_opp - 0.4)
            ax.grid(True, axis="x", color=GRID, lw=0.6, zorder=0)
            ax.tick_params(colors=MUTED, labelsize=8, length=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color("#c3c2b7")
            if r == 0:
                ax.set_title(f"f{fn}", fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(f"d = {dim}", fontsize=9, color=INK)
            if r == len(dims) - 1:
                ax.set_xlabel(f"r_rb   ← {reference} besser · schlechter →", fontsize=8.5,
                              color=INK)

    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    # Aufruf aus benchmark/: python -m postprocessing.wilcoxon_auc [csv] [metrik]
    csv = sys.argv[1] if len(sys.argv) > 1 else max((config.RESULTS_DIR / "raw").glob("results_*.csv"))
    metric = sys.argv[2] if len(sys.argv) > 2 else "simple_regret"

    runs = load_auc(csv, metric)
    out_dir = FIGURES_DIR / "auc" / Path(csv).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for scenario, sub in runs.groupby("scenario"):
        label = SCENARIO_LABELS.get(scenario, scenario)
        name = f"wilcoxon_auc_{scenario}_{metric}"
        table = wilcoxon_auc_table(sub)
        if table.empty:
            continue
        print(_save_table(table, f"{name}_{Path(csv).stem}"))
        # Holm bleibt ueber das ganze Szenario, ein Bild je Intensitaet
        for intensity, part in table.groupby("intensity"):
            print(plot_auc_bars(
                part,
                f"Wilcoxon-Vorzeichen-Rang-Test auf der Fläche unter der log-Regret-Kurve – "
                f"{REFERENCE} gegen alle Verfahren, {label}, Stufe {intensity:g}\n"
                f"{metric}, Mittel über {part['n_iterations'].max()} Iterationen, "
                f"n = {int(part['n'].max())} gepaarte Wiederholungen · Balken: rang-biseriale "
                f"Korrelation · Text: Median-Faktor {REFERENCE}/Gegner · {SIGNIFICANCE_NOTE}",
                out_dir / f"{name}_{intensity:g}.png"))
