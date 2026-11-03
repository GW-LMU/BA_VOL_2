# postprocessing/optimization_path_plot.py
# Zeigt den Optimierungspfad eines Laufs (2D) auf der BBOB-Funktion:
#   links:  3D-Oberflaeche von der Seite
#   rechts: Draufsicht als Karte, Hoehe ueber die Farbe
# Startpunkte gruen, danach angefragte Punkte (next_x) in Reihenfolge mit Linie verbunden.
# Aufruf aus dem Ordner benchmark/:
#     python postprocessing/optimization_path_plot.py --csv results/raw/results_2026-10-26_start10_budget100.csv --algorithm GP_BO --function 3
#     python postprocessing/optimization_path_plot.py --csv ... --task-id GP_BO_f3_d2_i1_nominal_level0_r00

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing import config
from preprocessing.coco_plots import BBOB_NAMES, evaluate_grid

OUT_DIR = config.RESULTS_DIR / "figures" / "optimierungspfad"

START_COLOR = "#2e9e5b"
# Pfad nach Iteration eingefaerbt: frueh hell (rosa), spaet dunkel (rot)
PATH_CMAP = LinearSegmentedColormap.from_list("pfad", ["#ffc2c2", "#d62728", "#5c0000"])


# Liest einen Lauf (alle Iterationen einer task_id) aus der Ergebnis-CSV
def load_run(csv_path, task_id=None, algorithm=None, function=None, scenario="nominal",
             intensity=0.0, repetition=0, instance=1):
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[(df["status"] == "ok") & (df["dimension"] == 2)]
    if task_id is None:
        df = df[(df["algorithm"] == algorithm) & (df["function"] == function)
                & (df["scenario"] == scenario) & np.isclose(df["intensity"], intensity)
                & (df["repetition"] == repetition) & (df["instance"] == instance)]
    else:
        df = df[df["task_id"] == task_id]
    if df.empty:
        raise ValueError("Kein passender 2D-Lauf in der CSV gefunden.")

    df = df.sort_values("iteration")
    first = df.iloc[0]
    run = {
        "task_id": first["task_id"],
        "algorithm": first["algorithm"],
        "function": int(first["function"]),
        "instance": int(first["instance"]),
        "start_X": np.array(json.loads(first["start_X"])),
        "start_f": np.array(json.loads(first["start_f_true"])),
        "path_X": np.array([json.loads(x) for x in df["next_x"]]),
        "path_f": df["next_f_true"].to_numpy(),
    }
    return run


def plot_path(run, n_grid=200, out_path=None):
    X, Y, Z = evaluate_grid(run["function"], run["instance"], n_grid)
    sx, sf = run["start_X"], run["start_f"]
    px, pf = run["path_X"], run["path_f"]

    # bester gefundener Punkt (Start + Pfad)
    all_X, all_f = np.vstack([sx, px]), np.concatenate([sf, pf])
    best = int(np.argmin(all_f))

    # Farbe pro Iteration; Segment i verbindet Punkt i mit i+1 und bekommt dessen Farbe
    iterations = np.arange(1, len(px) + 1)
    norm = Normalize(vmin=1, vmax=max(2, len(px)))
    seg_colors = PATH_CMAP(norm(iterations[1:]))

    fig = plt.figure(figsize=(17, 6.5))

    # 3D-Ansicht von der Seite (Oberflaeche transparent, damit der Pfad sichtbar bleibt)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.6)
    pts3d = np.column_stack([px, pf])
    ax3d.add_collection3d(Line3DCollection(np.stack([pts3d[:-1], pts3d[1:]], axis=1),
                                           colors=seg_colors, linewidth=1.2))
    ax3d.scatter(px[:, 0], px[:, 1], pf, c=iterations, cmap=PATH_CMAP, norm=norm, s=12,
                 depthshade=False, label="BO-Iterationen")
    ax3d.scatter(sx[:, 0], sx[:, 1], sf, color=START_COLOR, s=40, edgecolor="black",
                 depthshade=False, label="Startpunkte")
    ax3d.scatter(*all_X[best], all_f[best], color="gold", marker="*", s=250,
                 edgecolor="black", depthshade=False, label="bester Punkt")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.view_init(elev=30, azim=45)
    ax3d.legend(loc="upper left")

    # Draufsicht: Hoehe ueber die Farbe
    ax2d = fig.add_subplot(1, 2, 2)
    cont = ax2d.contourf(X, Y, Z, levels=50, cmap="viridis")
    fig.colorbar(cont, ax=ax2d, label="z")
    ax2d.add_collection(LineCollection(np.stack([px[:-1], px[1:]], axis=1),
                                       colors=seg_colors, linewidth=1.2))
    path_sc = ax2d.scatter(px[:, 0], px[:, 1], c=iterations, cmap=PATH_CMAP, norm=norm,
                           s=14, edgecolor="black", linewidth=0.3, zorder=2,
                           label="BO-Iterationen")
    fig.colorbar(path_sc, ax=ax2d, label="Iteration", pad=0.02)
    ax2d.scatter(sx[:, 0], sx[:, 1], color=START_COLOR, s=50, edgecolor="black",
                 zorder=3, label="Startpunkte")
    ax2d.scatter(*all_X[best], color="gold", marker="*", s=300, edgecolor="black",
                 zorder=4, label="bester Punkt")
    ax2d.annotate("Ende", px[-1], textcoords="offset points", xytext=(6, 6),
                  color="white", fontsize=9)
    ax2d.set_xlabel("x")
    ax2d.set_ylabel("y")
    ax2d.set_aspect("equal")
    ax2d.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"f{run['function']}: {BBOB_NAMES[run['function']]}  |  {run['task_id']}")
    fig.tight_layout()

    if out_path is None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"pfad_{run['task_id']}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot gespeichert unter {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--algorithm")
    parser.add_argument("--function", type=int)
    parser.add_argument("--scenario", default="nominal")
    parser.add_argument("--intensity", type=float, default=0.0)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--instance", type=int, default=1)
    parser.add_argument("--grid", type=int, default=200)
    args = parser.parse_args()

    if args.task_id is None and (args.algorithm is None or args.function is None):
        parser.error("entweder --task-id oder --algorithm und --function angeben")

    run = load_run(args.csv, args.task_id, args.algorithm, args.function, args.scenario,
                   args.intensity, args.repetition, args.instance)
    plot_path(run, args.grid)
