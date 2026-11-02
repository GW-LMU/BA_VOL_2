# postprocessing/
#
# Wilcoxon-Vorzeichen-Rang-Test in JEDER Iteration statt nur in der letzten.
#
# Pro Setting (Szenario x Intensitaet x Funktion x Dimension) und Iteration t
# wird das Referenzverfahren (PROBO) gegen jedes andere Verfahren getestet:
#     d_r(t) = log10(m_PROBO,r(t)) - log10(m_B,r(t))      (gepaart ueber Wiederholung r)
# Test, Pratt-Behandlung der Nulldifferenzen und r_rb wie in statistics.py.
#
# Darstellung als "vorzeichenbehafteter p-Wert":
#     s(t) = -log10(p(t)) * Richtung,  Richtung = +1 wenn PROBO besser (r_rb < 0)
# Oberhalb des grauen Bands ist PROBO signifikant besser, unterhalb signifikant
# schlechter, im Band kein signifikanter Unterschied (p >= 0.05).
#
# Mehrfachtests: Holm-Korrektur ueber die Vergleiche (Gegner) einer Iteration
# innerhalb eines Settings. Ueber die Iterationen wird NICHT korrigiert -
# die Verlaeufe sind beschreibend (wann tritt ein Unterschied auf?).

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing import config
from postprocessing.statistics import (FIGURES_DIR, INK, MUTED, SCENARIO_LABELS, _holm, _log,
                                       _save_table, _wilcoxon_diff)

REFERENCE = "PROBO"
OPPONENT_COLORS = {"GP_BO": "#2a78d6", "STABLEOPT": "#2e9e5b", "RP_BO": "#8a5cc2",
                   "DRBO": "#c2417a", "AIRBO": "#1a9fb3"}
ALPHA = 0.05
GRID = "#e1e0d9"


def load_all_iterations(csv_path, metric="simple_regret", scenarios=None):
    """Metrik in jeder Iteration jedes erfolgreichen Experiments."""
    cols = ["algorithm", "function", "dimension", "scenario", "intensity",
            "repetition", "iteration", "status", metric]
    parts = []
    # Die CSV kann mehrere GB gross sein -> stueckweise lesen und sofort filtern
    for chunk in pd.read_csv(csv_path, usecols=cols, chunksize=200_000):
        keep = chunk["status"] == "ok"
        if scenarios is not None:
            keep &= chunk["scenario"].isin(scenarios)
        parts.append(chunk[keep])
    return pd.concat(parts).drop(columns="status")


def wilcoxon_per_iteration(df, metric="simple_regret", reference=REFERENCE):
    """Referenz gegen jedes andere Verfahren, je Setting und Iteration."""
    rows = []
    setting_keys = ["scenario", "intensity", "function", "dimension"]
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
            for iteration in wide.index:
                a, b = ref.loc[iteration, reps], other.loc[iteration, reps]
                valid = a.notna() & b.notna()
                if valid.sum() < 2:
                    continue
                rows.append({**dict(zip(setting_keys, setting)), "iteration": iteration,
                             "comparison": f"{reference} vs. {opponent}", "A": reference,
                             "B": opponent,
                             **_wilcoxon_diff(_log(a[valid]) - _log(b[valid]))})
    table = pd.DataFrame(rows)
    table["p_holm"] = table.groupby(setting_keys + ["iteration"])["p"].transform(
        lambda p: _holm(p.values))
    # Vorzeichenbehafteter p-Wert: + = Referenz besser, - = Referenz schlechter
    table["signed_log_p"] = -np.log10(table["p_holm"].clip(lower=1e-300)) * -np.sign(table["r_rb"])
    return table


def _p_ticks(limit):
    """y-Ticks in -log10(p)-Einheiten, beschriftet mit dem p-Wert (beide Richtungen)."""
    levels = [(0.0, "1"), (-np.log10(ALPHA), f"{ALPHA:g}"), (3.0, "10⁻³")]
    levels += [(float(e), f"10⁻{str(e).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))}")
               for e in range(6, int(limit) + 1, 3)]
    ticks, labels = [], []
    for value, label in levels:
        ticks += [value] if value == 0 else [-value, value]
        labels += [label] if value == 0 else [label, label]
    order = np.argsort(ticks)
    return np.array(ticks)[order], np.array(labels)[order]


def plot_wilcoxon_over_iterations(table, scenario, intensity, out_path, reference=REFERENCE):
    """Ein Bild pro Szenario und Intensitaet: Zeilen = Dimension, Spalten = Funktion,
    je Panel eine Linie pro Gegner (vorzeichenbehafteter, Holm-korrigierter p-Wert)."""
    sub = table[(table["scenario"] == scenario) & (table["intensity"] == intensity)]
    functions = sorted(sub["function"].unique())
    dims = sorted(sub["dimension"].unique())
    opponents = [o for o in OPPONENT_COLORS if o in sub["B"].unique()]
    opponents += [o for o in sub["B"].unique() if o not in opponents]
    limit = max(3.5, np.ceil(sub["signed_log_p"].abs().max() + 0.3))
    threshold = -np.log10(ALPHA)

    fig, axes = plt.subplots(len(dims), len(functions), sharex=True, sharey=True,
                             figsize=(3.4 * len(functions) + 0.8, 2.5 * len(dims) + 1.0),
                             squeeze=False)
    for r, dim in enumerate(dims):
        for c, fn in enumerate(functions):
            ax = axes[r, c]
            panel = sub[(sub["dimension"] == dim) & (sub["function"] == fn)]
            ax.axhspan(-threshold, threshold, color="#efeee9", lw=0, zorder=0)
            ax.axhline(0, color="#c3c2b7", lw=0.8, zorder=1)
            for opponent in opponents:
                line = panel[panel["B"] == opponent].sort_values("iteration")
                if line.empty:
                    continue
                ax.plot(line["iteration"], line["signed_log_p"],
                        color=OPPONENT_COLORS.get(opponent, MUTED), lw=1.6,
                        label=f"vs. {opponent}", zorder=2)
            ax.set_ylim(-limit, limit)
            ax.grid(True, color=GRID, lw=0.6)
            ax.tick_params(colors=MUTED, labelsize=8, length=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#c3c2b7")
            if r == 0:
                ax.set_title(f"f{fn}", fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(f"d = {dim}\np-Wert (Holm)", fontsize=9, color=INK)
            if r == len(dims) - 1:
                ax.set_xlabel("Iteration", fontsize=9, color=INK)

    ticks, labels = _p_ticks(limit)
    axes[0, 0].set_yticks(ticks, labels)
    # Richtung einmal am oberen/unteren Rand des ersten Panels beschriften
    axes[0, 0].text(0.02, 0.97, f"↑ {reference} besser", transform=axes[0, 0].transAxes,
                    fontsize=8, color=MUTED, va="top")
    axes[0, 0].text(0.02, 0.03, f"↓ {reference} schlechter", transform=axes[0, 0].transAxes,
                    fontsize=8, color=MUTED, va="bottom")

    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(opponents), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    label = SCENARIO_LABELS.get(scenario, scenario)
    n_rep = int(sub["n"].max())
    fig.suptitle(f"Wilcoxon-Vorzeichen-Rang-Test je Iteration – {reference} gegen alle Verfahren, "
                 f"{label}, Stufe {intensity:g}, n = {n_rep} gepaarte Wiederholungen\n"
                 f"Graues Band: p ≥ {ALPHA:g} (kein signifikanter Unterschied) · "
                 f"Holm-korrigiert über die Vergleiche je Iteration", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    # Aufruf aus benchmark/: python -m postprocessing.wilcoxon_iterations [csv] [metrik]
    csv = sys.argv[1] if len(sys.argv) > 1 else max((config.RESULTS_DIR / "raw").glob("results_*.csv"))
    metric = sys.argv[2] if len(sys.argv) > 2 else "simple_regret"

    table = wilcoxon_per_iteration(load_all_iterations(csv, metric), metric)
    print(_save_table(table, f"wilcoxon_verlauf_{Path(csv).stem}_{metric}"))
    for (scenario, intensity), _ in table.groupby(["scenario", "intensity"]):
        print(plot_wilcoxon_over_iterations(
            table, scenario, intensity,
            FIGURES_DIR / "verlauf" / Path(csv).stem /
            f"wilcoxon_verlauf_{scenario}_{intensity:g}_{metric}.png"))
