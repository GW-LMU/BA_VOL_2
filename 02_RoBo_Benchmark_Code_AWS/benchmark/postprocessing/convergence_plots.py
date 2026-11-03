
# postprocessing/convergence_plots.py

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing import config

ALGO_COLORS = {"GP_BO": "#2a78d6", "PROBO": "#eb6834", "STABLEOPT": "#2e9e5b"}


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
# Optik "standard": Der Simple Regret streut ueber mehrere Groessenordnungen, daher
# wird auf der log10-Skala gemittelt: Linie = 10^mean(log10 r) (geometrisches Mittel),
# Band = 10^(mean +/- t_{0.975, n-1} * sd / sqrt(n)) ueber die Wiederholungen.
# Optik "rodemann": arithmetisches Mittel wie der Mean Optimization Path (MOP,
# Rodemann & Augustin 2024, Def. 6), KI = mean +/- t_{0.975, n-1} * sd / sqrt(n).

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
ROBUST_METRIC_LABELS = {
    "M1": "M1 Nominaler Regret",
    "M2": "M2 Erwartete Performance",
    "M3": "M3 Worst-Case-Performance",
    "M4": "M4 Shift-Performance",
    "M5": "M5 DRO-Performance",
    "M6": "M6 CVaR",
}
# M1-M6 liegen je Empfehlungsregel vor: _BO = best observed, _POM = posterior mean,
# _ROBUST = robuster Kandidat (nur STABLEOPT)
METRIC_LABELS = {
    "simple_regret": "Simple Regret",
    "instantaneous_regret": "Instantaneous Regret",
    "cumulative_regret": "Cumulative Regret",
    **{f"{m}_{rule}": f"{label} ({rule})" for rule in ("BO", "POM", "ROBUST")
       for m, label in ROBUST_METRIC_LABELS.items()},
}
FIGURES_DIR = config.RESULTS_DIR / "figures" / "konvergenz"

# Optik "rodemann": angelehnt an die Konvergenzplots in Rodemann & Augustin (2024)
# (ggplot-Stil: dunkelgraues Panel, weisses Gitter, Mittelwert mit Fehlerbalken
# pro Iteration statt Band). PROBO gruen, GP_BO hellgrau wie im Paper.
RODEMANN_COLORS = {"GP_BO": "#e6e6e6", "PROBO": "#1fa37f", "STABLEOPT": "#f2c14e"}
RODEMANN_PANEL = "#5e5e5e"


def _ci_curve(runs, metric, geometric=True):
    """Mittel und 95%-KI pro Iteration ueber die Wiederholungen.

    geometric=True: geometrisches Mittel (auf log10 gemittelt),
    geometric=False: arithmetisches Mittel = Mean Optimization Path (Rodemann).
    """
    curves = runs.pivot_table(index="iteration", columns="repetition", values=metric)
    if geometric:
        curves = np.log10(curves.clip(lower=1e-12))
    n = curves.notna().sum(axis=1)
    mean = curves.mean(axis=1)
    half = stats.t.ppf(0.975, n - 1) * curves.std(axis=1, ddof=1) / np.sqrt(n)
    if geometric:
        return curves.index.values, 10 ** mean, 10 ** (mean - half), 10 ** (mean + half)
    # untere KI-Grenze kann negativ werden -> fuer die log-Achse nach unten begrenzen
    return curves.index.values, mean, (mean - half).clip(lower=1e-12), mean + half


def _mean_label(optik):
    return "arithmetisches Mittel (MOP)" if optik == "rodemann" else "geometrisches Mittel"


def _style_axis(ax, optik="standard"):
    ax.set_yscale("log")
    if optik == "rodemann":
        # Arithmetisches KI reicht oft unter 0 -> Achse nicht an den Fehlerbalken,
        # sondern an den Mittelwertkurven ausrichten (auch ueber sharey-Nachbarn);
        # Balken mit unterer Grenze <= 0 enden dann am unteren Rand
        # (nur beschriftete Linien: die Kappen der Fehlerbalken sind ebenfalls Line2D)
        lines = [ln for a in ax.get_shared_y_axes().get_siblings(ax) for ln in a.get_lines()
                 if not ln.get_label().startswith("_")]
        ys = np.concatenate([np.asarray(ln.get_ydata(), dtype=float) for ln in lines] or [[]])
        lows = ys[ys > 0]
        if lows.size:
            ax.set_ylim(bottom=min(lows) / 3)
        ax.set_facecolor(RODEMANN_PANEL)
        ax.grid(True, which="major", color="white", lw=0.6, alpha=0.6)
        ax.grid(True, which="minor", axis="x", color="white", lw=0.3, alpha=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(colors="#3a3a3a", labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")


def _draw(ax, runs, metric, color, ls, label, optik="standard"):
    x, m, lo, hi = _ci_curve(runs, metric, geometric=optik != "rodemann")
    if optik == "rodemann":
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none", ecolor=color,
                    elinewidth=0.9, capsize=1.5, capthick=0.6, alpha=0.75)
        ax.plot(x, m, color=color, lw=1.6, label=label)
        return
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
    ax.plot(x, m, color=color, ls=ls, lw=2, label=label)


def _load(csv_path):
    # Nur vorhandene Spalten laden (aeltere CSVs haben noch M1-M6 ohne Regel-Suffix)
    wanted = {"algorithm", "function", "dimension", "scenario", "intensity",
              "repetition", "iteration", "status", *METRIC_LABELS}
    df = pd.read_csv(csv_path, usecols=lambda c: c in wanted)
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


def plot_convergence_by_function(csv_path, out_path=None, metric="simple_regret",
                                 scenario="nominal", intensity=0.0, optik="standard"):
    """Vergleich der Verfahren pro Funktion in einem Szenario.

    Zeilen = Dimensionen, Spalten = BBOB-Funktionen, Linien = Verfahren.
    """
    df = _load(csv_path)
    df = df[(df["scenario"] == scenario) & (df["intensity"] == intensity)]
    functions = sorted(df["function"].unique())
    dims = sorted(df["dimension"].unique())
    n_rep = df["repetition"].nunique()

    # sharey nur pro Zelle: f1 und f11 liegen Groessenordnungen auseinander
    fig, axes = plt.subplots(len(dims), len(functions), figsize=(4.2 * len(functions), 2.9 * len(dims)),
                             sharex=True, squeeze=False)
    for r, dim in enumerate(dims):
        for c, fn in enumerate(functions):
            ax = axes[r, c]
            sub = df[(df["dimension"] == dim) & (df["function"] == fn)]
            colors = RODEMANN_COLORS if optik == "rodemann" else ALGO_COLORS
            for algo, color in colors.items():
                runs = sub[sub["algorithm"] == algo]
                if not runs.empty:
                    _draw(ax, runs, metric, color, "-", algo, optik)
            _style_axis(ax, optik)
            if r == 0:
                ax.set_title(f"f{fn}", fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(f"d = {dim}\n{METRIC_LABELS[metric]} (log)", fontsize=9, color=INK)
            if r == len(dims) - 1:
                ax.set_xlabel("Iteration", fontsize=9, color=INK)
    axes[0, 0].legend(frameon=False, fontsize=8,
                      labelcolor="white" if optik == "rodemann" else MUTED)

    fig.suptitle(f"{METRIC_LABELS[metric]} pro Funktion – {SCENARIO_LABELS[scenario]} (Stufe {intensity:g})\n"
                 f"{_mean_label(optik)} über {n_rep} Wiederholungen, "
                 f"{'Fehlerbalken' if optik == 'rodemann' else 'Band'} = 95%-KI", fontsize=11, color=INK)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    optik_suffix = "_rodemann" if optik == "rodemann" else ""
    out_path = out_path or FIGURES_DIR / f"{metric}_dim_x_funktion_{scenario}_{intensity:g}{optik_suffix}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_convergence_comparison(csv_path, out_path=None, metric="simple_regret", function=None,
                                optik="standard"):
    """Vergleich der Verfahren pro Szenario und Stufe.

    Zeilen = Dimensionen, Spalten = Szenario-Stufen, Linien = Verfahren.
    function: nur diese BBOB-Funktion (sonst werden Funktionen pro Wiederholung gemittelt).
    """
    df = _load(csv_path)
    if function is not None:
        df = df[df["function"] == function]
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
            colors = RODEMANN_COLORS if optik == "rodemann" else ALGO_COLORS
            for algo, color in colors.items():
                runs = sub[sub["algorithm"] == algo]
                if not runs.empty:
                    _draw(ax, runs, metric, color, "-", algo, optik)
            _style_axis(ax, optik)
            if r == 0:
                ax.set_title(f"{SCENARIO_LABELS[scen]} (Stufe {level:g})", fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(f"d = {dim}\n{METRIC_LABELS[metric]} (log)", fontsize=9, color=INK)
            if r == len(dims) - 1:
                ax.set_xlabel("Iteration", fontsize=9, color=INK)
    axes[0, 0].legend(frameon=False, fontsize=8,
                      labelcolor="white" if optik == "rodemann" else MUTED)

    fn_label = f" – f{function}" if function is not None else ""
    algo_label = " vs. ".join(a for a in ALGO_COLORS if a in df["algorithm"].unique())
    fig.suptitle(f"{METRIC_LABELS[metric]}{fn_label}: {algo_label} pro Szenario – {_mean_label(optik)} über "
                 f"{df['repetition'].nunique()} Wiederholungen, "
                 f"{'Fehlerbalken' if optik == 'rodemann' else 'Band'} = 95%-Konfidenzintervall",
                 fontsize=11, color=INK)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fn_suffix = f"_f{function}" if function is not None else ""
    optik_suffix = "_rodemann" if optik == "rodemann" else ""
    out_path = out_path or FIGURES_DIR / f"{metric}_vergleich{fn_suffix}{optik_suffix}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    # Ohne Argument: neueste results_JJJJ-MM-TT.csv (Name sortiert = chronologisch)
    csv = sys.argv[1] if len(sys.argv) > 1 else max((config.RESULTS_DIR / "raw").glob("results_*.csv"))
    columns = pd.read_csv(csv, nrows=0).columns
    # _ROBUST-Spalten sind leer, solange kein Verfahren robuste Kandidaten liefert (STABLEOPT)
    filled = pd.read_csv(csv, usecols=lambda c: c.endswith("_ROBUST")).notna().any()
    for metric in [m for m in METRIC_LABELS if m in columns and filled.get(m, True)]:
        print(plot_convergence_per_algorithm(csv, metric=metric))
        print(plot_convergence_comparison(csv, metric=metric))
