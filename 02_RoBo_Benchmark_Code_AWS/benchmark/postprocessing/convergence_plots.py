
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


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else config.RESULTS_DIR / "raw" / "test_run.csv"
    print(plot_convergence(csv))
