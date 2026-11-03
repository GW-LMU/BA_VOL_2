# postprocessing/paired_difference.py
#
# Deskriptiver Verlauf des Unterschieds PROBO gegen jedes andere Verfahren:
# Effektstaerke mit Konfidenzband je Iteration statt eines p-Werts.
#
# Pro Setting (Szenario x Intensitaet x Funktion x Dimension), Gegner B und
# Iteration t, gepaart ueber Wiederholung r (gleiche Seeds, wie statistics.py):
#     d_r(t) = log10(m_PROBO,r(t)) - log10(m_B,r(t))
#
#   Median d(t)   -> d < 0: PROBO besser; -1 = eine Groessenordnung weniger Regret
#                    (10^median(d) ist der Faktor aus statistics.py)
#   95 %-Band     -> Perzentil-Bootstrap des Medians ueber die Wiederholungen.
#                    Es werden ganze Wiederholungen gezogen, dieselben fuer alle
#                    Iterationen, damit das Band den Verlauf eines Laufs erhaelt.
#   Gewinnrate(t) -> Anteil der Wiederholungen mit d_r(t) < 0, Gleichstand zaehlt
#                    halb. Gepaartes Gegenstueck zu Vargha-Delaneys A12
#                    (0.5 = kein Unterschied, 1 = PROBO in jeder Wiederholung besser).
#
# Die Baender sind beschreibend (wann und wie stark?). Inferenz ueber den
# ganzen Verlauf: wilcoxon_auc.py; je Iteration: wilcoxon_iterations.py.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing import config
from postprocessing.statistics import FIGURES_DIR, INK, MUTED, SCENARIO_LABELS, _log, _save_table
from postprocessing.wilcoxon_iterations import (GRID, OPPONENT_COLORS, REFERENCE,
                                                load_all_iterations)

N_BOOTSTRAP = 2000
CONFIDENCE = 0.95
SEED = 12345


def paired_difference_curves(df, metric="simple_regret", reference=REFERENCE,
                             n_boot=N_BOOTSTRAP, seed=SEED):
    """Median der gepaarten log10-Differenz mit Bootstrap-Band und Gewinnrate je Iteration."""
    rng = np.random.default_rng(seed)
    lo_q, hi_q = 100 * (1 - CONFIDENCE) / 2, 100 * (1 + CONFIDENCE) / 2
    setting_keys = ["scenario", "intensity", "function", "dimension"]
    tables = []
    for setting, sub in df.groupby(setting_keys):
        # Spalten = (Verfahren, Wiederholung), Zeilen = Iteration
        wide = sub.pivot_table(index="iteration", columns=["algorithm", "repetition"],
                               values=metric)
        algorithms = wide.columns.get_level_values(0).unique()
        if reference not in algorithms:
            continue
        ref = wide[reference]
        for opponent in [a for a in algorithms if a != reference]:
            other = wide[opponent]
            reps = ref.columns.intersection(other.columns)
            d = pd.DataFrame(_log(ref[reps]) - _log(other[reps]), index=wide.index)
            # Nur Wiederholungen, die bei beiden Verfahren ueber alle Iterationen vorliegen
            d = d.loc[:, d.notna().all()].to_numpy()
            if d.shape[1] < 2:
                continue
            idx = rng.integers(0, d.shape[1], size=(n_boot, d.shape[1]))
            boot = np.median(d[:, idx], axis=2)                  # Iteration x Bootstrap
            win = (d < 0).mean(axis=1) + 0.5 * (d == 0).mean(axis=1)
            win_boot = ((d[:, idx] < 0).mean(axis=2)
                        + 0.5 * (d[:, idx] == 0).mean(axis=2))
            tables.append(pd.DataFrame({
                **dict(zip(setting_keys, setting)), "iteration": wide.index,
                "comparison": f"{reference} vs. {opponent}", "A": reference, "B": opponent,
                "n": d.shape[1], "median_d": np.median(d, axis=1),
                "ci_low": np.percentile(boot, lo_q, axis=1),
                "ci_high": np.percentile(boot, hi_q, axis=1),
                "win_rate": win,
                "win_low": np.percentile(win_boot, lo_q, axis=1),
                "win_high": np.percentile(win_boot, hi_q, axis=1)}))
    return pd.concat(tables, ignore_index=True)


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")


def plot_curves(table, scenario, intensity, out_path, kind="difference", ci_style="band",
                reference=REFERENCE):
    """Ein Bild pro Szenario und Intensitaet: Zeilen = Dimension, Spalten = Funktion,
    je Panel Linie (Median bzw. Gewinnrate) und 95 %-Intervall pro Gegner.
    kind     = "difference" (log10-Differenz) oder "win_rate" (Gewinnrate)
    ci_style = "band" (durchgehendes Band) oder "errorbar" (senkrechter Strich mit
               Begrenzungsstrichen je Iteration; Gegner leicht seitlich versetzt)"""
    sub = table[(table["scenario"] == scenario) & (table["intensity"] == intensity)]
    functions = sorted(sub["function"].unique())
    dims = sorted(sub["dimension"].unique())
    opponents = [o for o in OPPONENT_COLORS if o in sub["B"].unique()]
    opponents += [o for o in sub["B"].unique() if o not in opponents]
    center, low, high = (("median_d", "ci_low", "ci_high") if kind == "difference"
                         else ("win_rate", "win_low", "win_high"))
    neutral = 0.0 if kind == "difference" else 0.5

    fig, axes = plt.subplots(len(dims), len(functions), sharex=True,
                             sharey=(kind == "win_rate"),
                             figsize=(3.4 * len(functions) + 0.8, 2.5 * len(dims) + 1.0),
                             squeeze=False)
    for r, dim in enumerate(dims):
        for c, fn in enumerate(functions):
            ax = axes[r, c]
            panel = sub[(sub["dimension"] == dim) & (sub["function"] == fn)]
            ax.axhline(neutral, color="#8f8e86", lw=0.9, zorder=1)
            for k, opponent in enumerate(opponents):
                line = panel[panel["B"] == opponent].sort_values("iteration")
                if line.empty:
                    continue
                color = OPPONENT_COLORS.get(opponent, MUTED)
                if ci_style == "errorbar":
                    # Versatz, damit sich die Striche der Gegner nicht verdecken
                    x = line["iteration"] + 0.3 * (k - (len(opponents) - 1) / 2)
                    ax.errorbar(x, line[center],
                                yerr=[line[center] - line[low], line[high] - line[center]],
                                fmt="none", ecolor=color, elinewidth=0.6, capsize=1.2,
                                capthick=0.6, alpha=0.6, zorder=2)
                    ax.plot(x, line[center], color=color, lw=1.2,
                            label=f"vs. {opponent}", zorder=3)
                else:
                    ax.fill_between(line["iteration"], line[low], line[high], color=color,
                                    alpha=0.18, lw=0, zorder=2)
                    ax.plot(line["iteration"], line[center], color=color, lw=1.6,
                            label=f"vs. {opponent}", zorder=3)
            if kind == "difference":
                # Symmetrisch um 0, damit "besser" und "schlechter" gleich viel Platz haben
                limit = max(0.5, np.nanmax(np.abs(panel[[low, high]].to_numpy())) * 1.05)
                ax.set_ylim(-limit, limit)
            else:
                ax.set_ylim(0, 1)
            _style(ax)
            if r == 0:
                ax.set_title(f"f{fn}", fontsize=10, color=INK)
            if c == 0:
                ylabel = "Δ log10 Regret" if kind == "difference" else "Gewinnrate"
                ax.set_ylabel(f"d = {dim}\n{ylabel}", fontsize=9, color=INK)
            if r == len(dims) - 1:
                ax.set_xlabel("Iteration", fontsize=9, color=INK)
            # Richtung in jedem Panel, da die y-Achsen der Differenz nicht geteilt sind
            better_top = kind == "win_rate"
            ax.text(0.98, 0.97 if better_top else 0.03, f"{reference} besser",
                    transform=ax.transAxes, fontsize=7, color=MUTED, ha="right",
                    va="top" if better_top else "bottom")
            ax.text(0.98, 0.03 if better_top else 0.97, f"{reference} schlechter",
                    transform=ax.transAxes, fontsize=7, color=MUTED, ha="right",
                    va="bottom" if better_top else "top")

    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(opponents), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    label = SCENARIO_LABELS.get(scenario, scenario)
    n_rep = int(sub["n"].max())
    ci_name = "Band" if ci_style == "band" else "Striche"
    if kind == "difference":
        title = (f"Gepaarte Differenz log10(Regret {reference}) − log10(Regret Gegner), "
                 f"{label}, Stufe {intensity:g}, n = {n_rep} Wiederholungen\n"
                 f"Linie: Median · {ci_name}: {CONFIDENCE:.0%}-Bootstrap-KI · "
                 f"−1 = {reference} hat eine Größenordnung weniger Regret")
    else:
        title = (f"Gewinnrate {reference}: Anteil der Wiederholungen mit kleinerem Regret, "
                 f"{label}, Stufe {intensity:g}, n = {n_rep} Wiederholungen\n"
                 f"{ci_name}: {CONFIDENCE:.0%}-Bootstrap-KI · 0.5 = kein Unterschied · "
                 f"Gleichstand zählt halb")
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    # Aufruf aus benchmark/: python -m postprocessing.paired_difference [csv] [metrik]
    csv = sys.argv[1] if len(sys.argv) > 1 else max((config.RESULTS_DIR / "raw").glob("results_*.csv"))
    metric = sys.argv[2] if len(sys.argv) > 2 else "simple_regret"

    table = paired_difference_curves(load_all_iterations(csv, metric), metric)
    print(_save_table(table, f"differenz_verlauf_{Path(csv).stem}_{metric}"))
    out_dir = FIGURES_DIR / "differenz" / Path(csv).stem
    for (scenario, intensity), _ in table.groupby(["scenario", "intensity"]):
        for kind, name in (("difference", "differenz"), ("win_rate", "gewinnrate")):
            for ci_style, suffix in (("band", ""), ("errorbar", "_fehlerbalken")):
                print(plot_curves(
                    table, scenario, intensity,
                    out_dir / f"{name}_{scenario}_{intensity:g}_{metric}{suffix}.png",
                    kind, ci_style))
