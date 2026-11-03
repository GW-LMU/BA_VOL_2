# postprocessing/statistics.py
#
# Gepaarte Wilcoxon-Vorzeichen-Rang-Tests auf den Endwerten (letzte Iteration).
#
# Paarung: Wiederholung r nutzt bei allen Verfahren und Szenarien dieselben
# Seeds (Startdesign, Prior, Rauschen; siehe preprocessing/seed_manager.py),
# daher wird pro Setting (Funktion x Dimension x Intensitaet) immer
# Wiederholung r mit Wiederholung r verglichen.
#
# Zwei Fragestellungen:
#   1. Verfahrensvergleich (wilcoxon_table): A gegen B im selben Szenario
#        d_r = log10(m_A,r) - log10(m_B,r)
#   2. Robustheitsverlust (robustness_loss_table): jedes Verfahren gegen sich
#      selbst im nominalen Fall, und der Verlust zweier Verfahren gegeneinander
#        Delta_r = log10(m_gestoert,r) - log10(m_nominal,r)
#        d_r     = Delta_A,r - Delta_B,r   (Interaktion Verfahren x Stoerung)
#
# Auswertung der Differenzen d (fuer beide gleich):
#   d < 0  ->  kleinerer Regret (besser) bzw. kleinerer Verlust
#   Faktor = 10^median(d), z. B. 0.3 = im Median 30 % des Regrets der Referenz
#   r_rb   = (W+ - W-) / (W+ + W-) in [-1, 1], Rang-biseriale Korrelation
#            (-1 = in jeder Wiederholung besser, +1 = in jeder schlechter)
# Mehrfachtests: Holm-Korrektur ueber alle Tests einer Abbildung.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing import config

PAIRS = [("GP_BO", "PROBO"), ("GP_BO", "STABLEOPT"), ("STABLEOPT", "PROBO")]
ALGORITHMS = ["GP_BO", "PROBO", "STABLEOPT"]
# Interaktion: Verliert das erste Verfahren durch die Stoerung mehr als das zweite?
LOSS_PAIRS = [("PROBO", "GP_BO"), ("PROBO", "STABLEOPT")]
SCENARIO_LABELS = {"nominal": "Nominal", "prior_error": "Prior-Fehler",
                   "stochastic_input": "Stochastischer Input"}
FIGURES_DIR = config.RESULTS_DIR / "figures" / "statistik"
TABLES_DIR = config.RESULTS_DIR / "tables"

# Divergierend: blau = besser, rot = schlechter, grau = kein Unterschied
EFFECT_CMAP = LinearSegmentedColormap.from_list("effect", ["#2a78d6", "#f0efec", "#e34948"])
INK, MUTED = "#0b0b0b", "#52514e"


def load_final_values(csv_path, metric="simple_regret", scenarios=("prior_error",)):
    """Wert der Metrik in der letzten Iteration jedes erfolgreichen Experiments."""
    if isinstance(scenarios, str):
        scenarios = (scenarios,)
    cols = ["task_id", "algorithm", "function", "dimension", "scenario", "intensity",
            "repetition", "iteration", "status", metric]
    parts = []
    # Die CSV kann mehrere GB gross sein -> stueckweise lesen und sofort filtern
    for chunk in pd.read_csv(csv_path, usecols=cols, chunksize=200_000):
        parts.append(chunk[(chunk["status"] == "ok") & chunk["scenario"].isin(scenarios)])
    df = pd.concat(parts)
    last = df.groupby("task_id")["iteration"].transform("max")
    return df[df["iteration"] == last].drop(columns=["status", "iteration"])


def _log(values):
    return np.log10(np.clip(np.asarray(values, dtype=float), 1e-12, None))


def _wilcoxon_diff(d):
    """Wilcoxon-Vorzeichen-Rang-Test auf gepaarten log10-Differenzen d (H0: Median 0)."""
    d = np.asarray(d, dtype=float)
    nonzero = d[d != 0]
    result = {"n": d.size, "n_ties": int(d.size - nonzero.size),
              "wins": int((d < 0).sum()), "losses": int((d > 0).sum()),
              "factor": float(10 ** np.median(d)),
              "W_plus": np.nan, "r_rb": 0.0, "p": 1.0}
    if nonzero.size == 0:
        return result
    ranks = stats.rankdata(np.abs(nonzero))
    w_plus, w_minus = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
    # Nulldifferenzen (gleicher empfohlener Punkt) nach Pratt: gehen in die
    # Rangbildung ein, zaehlen aber zu keiner Seite
    p = stats.wilcoxon(d, zero_method="pratt", alternative="two-sided").pvalue
    result.update(W_plus=float(w_plus), r_rb=float((w_plus - w_minus) / (w_plus + w_minus)),
                  p=float(p))
    return result


def _holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def wilcoxon_table(final, metric="simple_regret", pairs=PAIRS):
    """Verfahrensvergleich A gegen B innerhalb eines Szenarios."""
    rows = []
    for (intensity, fn, dim), sub in final.groupby(["intensity", "function", "dimension"]):
        wide = sub.pivot_table(index="repetition", columns="algorithm", values=metric)
        for a, b in pairs:
            if a not in wide or b not in wide:
                continue
            paired = wide[[a, b]].dropna()
            rows.append({"comparison": f"{a} vs. {b}", "intensity": intensity,
                         "function": fn, "dimension": dim, "A": a, "B": b,
                         **_wilcoxon_diff(_log(paired[a]) - _log(paired[b]))})
    table = pd.DataFrame(rows)
    table["p_holm"] = _holm(table["p"])
    return table


def robustness_loss_table(final, metric="simple_regret", scenario="prior_error",
                          algorithms=ALGORITHMS, loss_pairs=LOSS_PAIRS):
    """Robustheitsverlust Delta = log10(gestoert) - log10(nominal) je Verfahren
    und Differenz der Verluste zweier Verfahren (Interaktion)."""
    nominal = final[final["scenario"] == "nominal"]
    disturbed = final[final["scenario"] == scenario]
    key = ["algorithm", "function", "dimension", "repetition"]
    merged = disturbed.merge(nominal[key + [metric]], on=key, suffixes=("", "_nominal"))
    merged["delta"] = _log(merged[metric]) - _log(merged[f"{metric}_nominal"])

    rows = []
    for (intensity, fn, dim), sub in merged.groupby(["intensity", "function", "dimension"]):
        delta = sub.pivot_table(index="repetition", columns="algorithm", values="delta")
        setting = {"intensity": intensity, "function": fn, "dimension": dim}
        for algo in algorithms:
            if algo in delta:
                rows.append({"comparison": f"Δ {algo}", **setting, "A": algo, "B": "nominal",
                             **_wilcoxon_diff(delta[algo].dropna())})
        for a, b in loss_pairs:
            if a in delta and b in delta:
                paired = delta[[a, b]].dropna()
                rows.append({"comparison": f"Δ {a} − Δ {b}", **setting, "A": a, "B": b,
                             **_wilcoxon_diff(paired[a] - paired[b])})
    table = pd.DataFrame(rows)
    table["p_holm"] = _holm(table["p"])
    return table


def _format_factor(v):
    # Ab 10 ganze Zahlen mit Tausenderpunkt statt Exponentenschreibweise
    return f"{v:,.0f}".replace(",", ".") if v >= 10 else f"{v:.2g}"


def _stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def plot_wilcoxon_heatmaps(table, title, colorbar_label, out_path):
    """Zeilen = Intensitaeten, Spalten = Vergleiche; Zelle: x = Funktion, y = Dimension.

    Farbe = r_rb, Text = Median-Faktor und Holm-korrigierte Signifikanz.
    """
    intensities = sorted(table["intensity"].unique())
    comparisons = list(dict.fromkeys(table["comparison"]))
    functions = sorted(table["function"].unique())
    dims = sorted(table["dimension"].unique())

    fig, axes = plt.subplots(len(intensities), len(comparisons),
                             figsize=(3.6 * len(comparisons) + 1.2, 3.0 * len(intensities) + 0.9),
                             squeeze=False)
    for r, level in enumerate(intensities):
        for c, comparison in enumerate(comparisons):
            ax = axes[r, c]
            sub = table[(table["intensity"] == level) & (table["comparison"] == comparison)]
            grid = sub.pivot(index="dimension", columns="function", values="r_rb").reindex(
                index=dims, columns=functions)
            image = ax.imshow(grid.values, cmap=EFFECT_CMAP, vmin=-1, vmax=1, origin="lower",
                              aspect="auto")
            for row in sub.itertuples():
                x, y = functions.index(row.function), dims.index(row.dimension)
                significant = row.p_holm < 0.05
                ax.text(x, y, f"×{_format_factor(row.factor)}\n{_stars(row.p_holm)}",
                        ha="center", va="center", fontsize=9, color=INK if significant else MUTED,
                        fontweight="bold" if significant else "normal")
            # 2px-Spalt zwischen den Zellen in Hintergrundfarbe
            ax.set_xticks(np.arange(len(functions) + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(len(dims) + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="white", lw=2)
            ax.tick_params(which="minor", length=0)
            ax.set_xticks(range(len(functions)), [f"f{f}" for f in functions])
            ax.set_yticks(range(len(dims)), [f"d = {d}" for d in dims] if c == 0 else [])
            ax.tick_params(colors=MUTED, labelsize=9, length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(comparison, fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(f"Stufe {level:g}\nDimension", fontsize=9, color=INK)
            if r == len(intensities) - 1:
                ax.set_xlabel("BBOB-Funktion", fontsize=9, color=INK)

    bar = fig.colorbar(image, ax=axes, shrink=0.8, pad=0.02)
    bar.set_label(colorbar_label, fontsize=9, color=INK)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    bar.outline.set_visible(False)

    fig.suptitle(title, fontsize=10, color=INK)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_table(table, name):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / f"{name}.csv"
    # utf-8-sig, damit Excel das "Δ" in der Spalte comparison richtig anzeigt
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return path


SIGNIFICANCE_NOTE = "Holm-korrigiert (* p<0.05, ** p<0.01, *** p<0.001)"


if __name__ == "__main__":
    # Aufruf aus benchmark/: python -m postprocessing.statistics [csv] [metrik] [szenario]
    # (nicht als Skriptpfad, sonst verdeckt diese Datei das Standardmodul "statistics")
    csv = sys.argv[1] if len(sys.argv) > 1 else max((config.RESULTS_DIR / "raw").glob("results_*.csv"))
    metric = sys.argv[2] if len(sys.argv) > 2 else "simple_regret"
    scenario = sys.argv[3] if len(sys.argv) > 3 else "prior_error"
    label = SCENARIO_LABELS.get(scenario, scenario)

    final = load_final_values(csv, metric, ("nominal", scenario))
    n_rep = final["repetition"].nunique()

    # 1. Verfahrensvergleich innerhalb des Szenarios
    name = f"wilcoxon_{scenario}_{metric}"
    table = wilcoxon_table(final[final["scenario"] == scenario], metric)
    print(_save_table(table, name))
    print(plot_wilcoxon_heatmaps(
        table,
        f"Wilcoxon-Vorzeichen-Rang-Test – {label}, {metric} in der letzten Iteration, "
        f"n = {n_rep} gepaarte Wiederholungen\nZelle: Median-Faktor A/B des Regrets · "
        f"{SIGNIFICANCE_NOTE}",
        "Rang-biseriale Korrelation r\n← erstes Verfahren besser · schlechter →",
        FIGURES_DIR / f"{name}.png"))

    # 2. Robustheitsverlust gegenueber nominal
    name = f"robustheitsverlust_{scenario}_{metric}"
    table = robustness_loss_table(final, metric, scenario)
    print(_save_table(table, name))
    print(plot_wilcoxon_heatmaps(
        table,
        f"Robustheitsverlust durch {label} – {metric} in der letzten Iteration, "
        f"n = {n_rep} gepaarte Wiederholungen\nΔ = log10(Regret {label}) − log10(Regret nominal); "
        f"Zelle: Median-Faktor 10^Δ · {SIGNIFICANCE_NOTE}",
        "Rang-biseriale Korrelation r\n← Regret sinkt / kleinerer Verlust · "
        "Regret steigt / größerer Verlust →",
        FIGURES_DIR / f"{name}.png"))
