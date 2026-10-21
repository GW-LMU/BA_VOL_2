
# postprocessing/convergence_plots.py

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing import config

ALGO_COLORS = {"GP_BO": "#2a78d6", "PROBO": "#eb6834"}


def plot_convergence(csv_path, out_path=None, metric="simple_regret"):

    df = pd.read_csv(csv_path)
    df = df[df["status"] == "ok"]

    panels = (
        df[["scenario", "intensity"]]
        .drop_duplicates()
        .sort_values(["scenario", "intensity"], key=lambda s: s.map(
            {"nominal": 0, "prior_error": 1, "stochastic_input": 2}) if s.name == "scenario" else s)
        .itertuples(index=False)
    )
    panels = list(panels)

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.6), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (scenario, intensity) in zip(axes, panels):
        sub = df[(df["scenario"] == scenario) & (df["intensity"] == intensity)]

        for algorithm, color in ALGO_COLORS.items():
            runs = sub[sub["algorithm"] == algorithm]
            if runs.empty:
                continue
            curves = runs.pivot_table(index="iteration", columns="repetition", values=metric)
            # einzelne Wiederholungen dünn, Mittelwert kräftig
            for rep in curves.columns:
                ax.plot(curves.index, curves[rep].clip(lower=1e-6), color=color, lw=0.8, alpha=0.3)
            ax.plot(curves.index, curves.mean(axis=1).clip(lower=1e-6), color=color, lw=2, label=algorithm)

        ax.set_yscale("log")
        ax.set_title(f"{scenario} (Stufe {intensity:g})", fontsize=10)
        ax.set_xlabel("Iteration")
        ax.grid(True, which="major", color="#e5e5e5", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel(f"{metric} (log)")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Konvergenz – Mittelwert über Wiederholungen (dünn: einzelne Läufe)", fontsize=11)
    fig.tight_layout()

    if out_path is None:
        out_path = config.RESULTS_DIR / "figures" / f"convergence_{Path(csv_path).stem}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------
# Konvergenz mit 95%-Konfidenzintervallen
# ----------------------------------------------------------------------------
# Der Simple Regret streut ueber mehrere Groessenordnungen, daher wird auf der
# log10-Skala gemittelt: Linie = 10^mean(log10 r) (geometrisches Mittel),
# Band = 10^(mean +/- t_{0.975, n-1} * sd / sqrt(n)) ueber die Wiederholungen.

import numpy as np
from scipy import stats

SCENARIO_ORDER = ["nominal", "prior_error", "stochastic_input"]
SCENARIO_LABELS = {"nominal": "Nominal", "prior_error": "Prior-Fehler",
                   "stochastic_input": "Stochastischer Input"}
# Intensitaetsstufen pro Verfahren: nominal = grauer Referenzfall,
# Stoerstufen als geordnete Blau-Stufen (hell = schwach, dunkel = stark)
LEVEL_STYLES = [("#3987e5", "-"), ("#104281", "-")]
NOMINAL_STYLE = ("#898781", "--")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

# Spalte in results.csv -> Achsenbeschriftung (M1-M6 siehe postprocessing/metrics.py)
METRIC_LABELS = {
    "simple_regret": "Simple Regret",
    "instantaneous_regret": "Instantaneous Regret",
    "cumulative_regret": "Cumulative Regret",
    "M1": "M1 Nominaler Regret",
    "M2": "M2 Erwartete Performance",
    "M3": "M3 Worst-Case-Performance",
    "M4": "M4 Shift-Performance",
    "M5": "M5 DRO-Performance",
    "M6": "M6 CVaR",
}
FIGURES_DIR = config.RESULTS_DIR / "figures" / "konvergenz"


def _ci_curve(runs, metric):
    """Geometrisches Mittel und 95%-KI pro Iteration ueber die Wiederholungen."""
    log = np.log10(runs.pivot_table(index="iteration", columns="repetition",
                                    values=metric).clip(lower=1e-12))
    n = log.notna().sum(axis=1)
    mean = log.mean(axis=1)
    half = stats.t.ppf(0.975, n - 1) * log.std(axis=1, ddof=1) / np.sqrt(n)
    return log.index.values, 10 ** mean, 10 ** (mean - half), 10 ** (mean + half)


def _style_axis(ax):
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")


def _draw(ax, runs, metric, color, ls, label):
    x, m, lo, hi = _ci_curve(runs, metric)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
    ax.plot(x, m, color=color, ls=ls, lw=2, label=label)


def _load(csv_path):
    df = pd.read_csv(csv_path, usecols=["algorithm", "dimension", "scenario", "intensity",
                                        "repetition", "iteration", "status",
                                        *METRIC_LABELS])
    return df[df["status"] == "ok"]


def plot_convergence_per_algorithm(csv_path, out_path=None, metric="simple_regret"):
    """Pro Verfahren: Wirkung der Stoerstufen je Szenario, nominal als Referenz.

    Zeilen = Dimensionen, Spalten = Verfahren x Stoerszenario.
    """
    df = _load(csv_path)
    algorithms = [a for a in ALGO_COLORS if a in df["algorithm"].unique()]
    scenarios = [s for s in SCENARIO_ORDER[1:] if s in df["scenario"].unique()]
    dims = sorted(df["dimension"].unique())
    cols = [(a, s) for a in algorithms for s in scenarios]

    fig, axes = plt.subplots(len(dims), len(cols), figsize=(3.6 * len(cols), 2.9 * len(dims)),
                             sharex=True, sharey="row", squeeze=False)
    for r, dim in enumerate(dims):
        for c, (algo, scen) in enumerate(cols):
            ax = axes[r, c]
            sub = df[(df["algorithm"] == algo) & (df["dimension"] == dim)]
            nominal = sub[sub["scenario"] == "nominal"]
            if not nominal.empty:
                _draw(ax, nominal, metric, *NOMINAL_STYLE, "nominal (Stufe 0)")
            levels = sorted(sub.loc[sub["scenario"] == scen, "intensity"].unique())
            for (color, ls), level in zip(LEVEL_STYLES, levels):
                _draw(ax, sub[(sub["scenario"] == scen) & (sub["intensity"] == level)],
                      metric, color, ls, f"Stufe {level:g}")
            _style_axis(ax)
            if r == 0:
                ax.set_title(f"{algo} – {SCENARIO_LABELS[scen]}", fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(f"d = {dim}\n{METRIC_LABELS[metric]} (log)", fontsize=9, color=INK)
            if r == len(dims) - 1:
                ax.set_xlabel("Iteration", fontsize=9, color=INK)
            if r == 0:
                ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)

    fig.suptitle(f"{METRIC_LABELS[metric]} pro Verfahren und Szenario – geometrisches Mittel über "
                 "10 Wiederholungen, Band = 95%-Konfidenzintervall", fontsize=11, color=INK)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out_path or FIGURES_DIR / f"{metric}_pro_verfahren.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_convergence_comparison(csv_path, out_path=None, metric="simple_regret"):
    """Vergleich der Verfahren pro Szenario und Stufe.

    Zeilen = Dimensionen, Spalten = Szenario-Stufen, Linien = Verfahren.
    """
    df = _load(csv_path)
    panels = (df[["scenario", "intensity"]].drop_duplicates()
              .assign(o=lambda d: d["scenario"].map(SCENARIO_ORDER.index))
              .sort_values(["o", "intensity"])[["scenario", "intensity"]]
              .itertuples(index=False))
    panels = list(panels)
    dims = sorted(df["dimension"].unique())

    fig, axes = plt.subplots(len(dims), len(panels), figsize=(3.4 * len(panels), 2.9 * len(dims)),
                             sharex=True, sharey="row", squeeze=False)
    for r, dim in enumerate(dims):
        for c, (scen, level) in enumerate(panels):
            ax = axes[r, c]
            sub = df[(df["dimension"] == dim) & (df["scenario"] == scen) & (df["intensity"] == level)]
            for algo, color in ALGO_COLORS.items():
                runs = sub[sub["algorithm"] == algo]
                if not runs.empty:
                    _draw(ax, runs, metric, color, "-", algo)
            _style_axis(ax)
            if r == 0:
                ax.set_title(f"{SCENARIO_LABELS[scen]} (Stufe {level:g})", fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(f"d = {dim}\n{METRIC_LABELS[metric]} (log)", fontsize=9, color=INK)
            if r == len(dims) - 1:
                ax.set_xlabel("Iteration", fontsize=9, color=INK)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=MUTED)

    fig.suptitle(f"{METRIC_LABELS[metric]}: GP_BO vs. PROBO pro Szenario – geometrisches Mittel über "
                 "10 Wiederholungen, Band = 95%-Konfidenzintervall", fontsize=11, color=INK)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out_path or FIGURES_DIR / f"{metric}_vergleich.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else config.RESULTS_DIR / "raw" / "results.csv"
    for metric in METRIC_LABELS:
        print(plot_convergence_per_algorithm(csv, metric=metric))
        print(plot_convergence_comparison(csv, metric=metric))
