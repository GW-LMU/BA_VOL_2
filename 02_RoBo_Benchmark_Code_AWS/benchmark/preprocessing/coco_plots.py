# preprocessing/coco_plots.py
# Plottet alle 24 Funktionen der BBOB-Suite in 2D als 3D-Oberflaeche (Stil wie Ackley-Plot).
# Aufruf aus dem Ordner benchmark/:
#     python preprocessing/coco_plots.py              # Instanz 1, Gitter 200x200
#     python preprocessing/coco_plots.py --instance 2 --grid 300

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing import config
from preprocessing.coco_setup import evaluate_bbob_function, get_problem

OUT_DIR = config.RESULTS_DIR / "figures" / "coco_plots"

BBOB_NAMES = {
    1: "Sphere", 2: "Ellipsoidal", 3: "Rastrigin", 4: "Bueche-Rastrigin",
    5: "Linear Slope", 6: "Attractive Sector", 7: "Step Ellipsoidal",
    8: "Rosenbrock", 9: "Rosenbrock (rotiert)", 10: "Ellipsoidal (hoch kond.)",
    11: "Discus", 12: "Bent Cigar", 13: "Sharp Ridge", 14: "Different Powers",
    15: "Rastrigin (multimodal)", 16: "Weierstrass", 17: "Schaffers F7",
    18: "Schaffers F7 (schlecht kond.)", 19: "Griewank-Rosenbrock F8F2",
    20: "Schwefel", 21: "Gallagher 101 Peaks", 22: "Gallagher 21 Peaks",
    23: "Katsuura", 24: "Lunacek bi-Rastrigin",
}


# Wertet die Funktion auf einem regelmaessigen Gitter ueber den Suchraum aus
def evaluate_grid(funktion, instance=1, n_grid=200):
    problem = get_problem(funktion, 2, instance)
    lb, ub = problem.lower_bounds, problem.upper_bounds
    x = np.linspace(lb[0], ub[0], n_grid)
    y = np.linspace(lb[1], ub[1], n_grid)
    X, Y = np.meshgrid(x, y)
    Z = np.array([evaluate_bbob_function([xi, yi], problem)
                  for xi, yi in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
    problem.free()
    return X, Y, Z


def plot_surface(ax, X, Y, Z, title, fig=None):
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True,
                           rstride=1, cstride=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.view_init(elev=30, azim=45)
    if fig is not None:
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)


def main(instance=1, n_grid=200):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Zwei Uebersichten: f1-f12 und f13-f24
    overviews = [plt.subplots(3, 4, figsize=(20, 15), subplot_kw={"projection": "3d"})
                 for _ in range(2)]

    for funktion in range(1, 25):
        X, Y, Z = evaluate_grid(funktion, instance, n_grid)
        title = f"f{funktion}: {BBOB_NAMES[funktion]}"

        # Einzelplot
        fig = plt.figure(figsize=(7, 5.5))
        ax = fig.add_subplot(projection="3d")
        plot_surface(ax, X, Y, Z, title, fig)
        fig.savefig(OUT_DIR / f"bbob_f{funktion:02d}_2d_i{instance}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Uebersicht (gröberes Gitter reicht)
        step = max(1, n_grid // 80)
        _, axes = overviews[(funktion - 1) // 12]
        plot_surface(axes.flat[(funktion - 1) % 12], X[::step, ::step], Y[::step, ::step],
                     Z[::step, ::step], title)
        print(f"f{funktion} geplottet.")

    for teil, (overview, _) in enumerate(overviews):
        von, bis = 12 * teil + 1, 12 * teil + 12
        overview.tight_layout()
        overview.savefig(OUT_DIR / f"bbob_f{von:02d}-f{bis:02d}_2d_i{instance}.png",
                         dpi=120, bbox_inches="tight")
        plt.close(overview)
    print(f"Plots gespeichert unter {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, default=1)
    parser.add_argument("--grid", type=int, default=200)
    args = parser.parse_args()
    main(args.instance, args.grid)
