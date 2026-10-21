"""BO-Kernfunktionen fuer die verteilte Pipeline.

Eigenstaendige Kopie/Anpassung der Logik aus ROBO/coco_benchmark.py und
ROBO/Standard_PY.py (dort nicht veraendert), damit ROBO_NEU ohne Abhaengigkeit
zum alten Ordner auf jede LRZ-Instanz kopiert werden kann. Die Standard-GP-
Variante liegt in gaussian_process.py, die weiteren robusten Varianten
(Imprecise GP, Relevance Pursuit, STABLEOPT, DRBO) sind analog angepasste
Kopien aus ROBO/, AIRBO ist eine schlanke Neuimplementierung nach Yang et al.
2023 (ROBO/AIRBO.py selbst war leer, siehe airbo.py fuer Details/Annahmen).
Alle sechs werden unten in select_and_run_variant() angebunden.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pfade  # noqa: F401 - fuegt Pre_Processing/ (config) und BO_Verfahren/ zu sys.path hinzu

import json
from functools import lru_cache

import numpy as np
import scipy.optimize
import torch

import config


def evaluate_bbob_function(x, funktion, dimension, suite, instance=1):# ✅
    """Negierter BBOB-Funktionswert: -f(x). BBOB-Probleme sind als
    Minimierung von f(x) definiert, mit dem bekannten (analytischen)
    Optimum im Inneren des Suchraums. Da hier durchgaengig MAXIMIERT wird
    (BoTorch-Konvention), wird die Standard-BBOB-Konvention hergestellt,
    indem -f(x) statt des rohen f(x) zurueckgegeben wird - das Maximum von
    -f(x) entspricht dann exakt dem Minimum von f(x)."""
    problem = suite.get_problem_by_function_dimension_instance(
        function=funktion, dimension=dimension, instance=instance
    )
    x_numpy = np.asarray(x, dtype=float).reshape(-1)
    y_val = problem(x_numpy)
    return -float(y_val)


def compute_regret(X_history, Y_history, x_opt, f_opt):# ✅
    y_values = np.asarray(Y_history, dtype=float)
    x_opt = np.asarray(x_opt, dtype=float)

    x_distances = np.array([np.linalg.norm(np.asarray(x, dtype=float).reshape(-1) - x_opt) for x in X_history])
    best_so_far = np.maximum.accumulate(y_values)
    simple_regret = f_opt - best_so_far
    immediate_regret = f_opt - y_values

    return best_so_far, simple_regret, immediate_regret, x_distances


# --- Kennzahlen am empfohlenen Punkt (rauschfreies f) ------------------------

def _sample_context(rng, n, lower_bound, upper_bound):
    """Zieht n Kontexte aus P* = N(CONTEXT_MEAN, CONTEXT_SIGMA^2), gestutzt auf
    [lower_bound, upper_bound] (Verwerfungsverfahren, kein Clipping - Clipping
    wuerde Masse auf den Rand legen)."""
    out = np.empty(0)
    while out.size < n:
        draw = rng.normal(config.CONTEXT_MEAN, config.CONTEXT_SIGMA, size=2 * n)
        out = np.concatenate([out, draw[(draw >= lower_bound) & (draw <= upper_bound)]])
    return out[:n]


@lru_cache(maxsize=None)
def _metric_sets(dim, lower_bound, upper_bound):
    """Feste Stoerungsmengen fuer compute_all_metrics, einmal pro Dimension aus
    config.METRIC_SEED gezogen - fuer ALLE Verfahren und Tasks identisch und
    unabhaengig von den Rausch-RNGs.

    ball:     Stoerungen delta mit ||delta|| <= METRIC_EPSILON: Mittelpunkt, die
              2*dim Achsenpunkte auf dem Rand, Rest je zur Haelfte auf der
              Kugeloberflaeche (dort liegt das Minimum glatter Funktionen
              haeufig) und gleichverteilt im Inneren.
    gauss:    eps ~ N(0, METRIC_SIGMA_X^2 I) fuer robust_expected.
    contexts: feste Stichprobe aus P* fuer die Kontext-Kennzahlen.
    """
    rng = np.random.default_rng([config.METRIC_SEED, dim])
    eps = config.METRIC_EPSILON

    axes = np.vstack([np.eye(dim), -np.eye(dim)]) * eps
    n_rest = max(config.METRIC_N_WORSTCASE - 1 - axes.shape[0], 0)
    directions = rng.normal(size=(n_rest, dim))
    directions /= np.clip(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12, None)
    radii = np.ones((n_rest, 1)) * eps
    n_inner = n_rest // 2
    radii[:n_inner] = eps * rng.uniform(0.0, 1.0, size=(n_inner, 1)) ** (1.0 / dim)
    ball = np.vstack([np.zeros((1, dim)), axes, directions * radii])

    gauss = rng.normal(0.0, config.METRIC_SIGMA_X, size=(config.METRIC_N_EXPECTED, dim))
    contexts = _sample_context(rng, config.METRIC_N_CONTEXT, lower_bound, upper_bound)
    return ball, gauss, contexts


def _evaluate_many(f, X):
    return np.array([f(x) for x in X])


def _robust_worstcase(f, x, lower_bound, upper_bound):
    ball, _, _ = _metric_sets(len(x), lower_bound, upper_bound)
    return float(_evaluate_many(f, np.clip(x + ball, lower_bound, upper_bound)).min())


def _context_worstcase(values, contexts):
    """Schlechtester Erwartungswert ueber alle Verteilungen w auf der festen
    Kontextstichprobe mit MMD(w, gleichverteilt) <= METRIC_CONTEXT_EPSILON
    (inneres Problem aus Kirschner et al. 2020, hier mit dem WAHREN f statt
    ucb). NaN, wenn der Loeser nicht konvergiert - bewusst kein stiller
    Rueckfall auf den nicht-robusten Mittelwert."""
    n = len(values)
    diffs = contexts[:, None] - contexts[None, :]
    M = np.exp(-0.5 * diffs ** 2 / config.METRIC_CONTEXT_MMD_LENGTHSCALE ** 2)
    w0 = np.full(n, 1.0 / n)
    eps2 = config.METRIC_CONTEXT_EPSILON ** 2
    res = scipy.optimize.minimize(
        lambda w: w @ values, w0.copy(), jac=lambda w: values, method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w: eps2 - (w - w0) @ M @ (w - w0)},
        ],
    )
    return float(res.x @ values) if res.success else float("nan")


def compute_all_metrics(f, x_rec, lower_bound, upper_bound):
    """Kennzahlen am empfohlenen Punkt x_rec, alle mit dem rauschfreien f
    (Maximierungs-Konvention -f, wie evaluate_bbob_function):

      f_true_rec        f(x_rec)
      robust_worstcase  min_{||delta|| <= eps} f(x_rec + delta)          (StableOpt-Ziel)
      robust_expected   E[f(x_rec + eps)], eps ~ N(0, sigma^2 I)           (AIRBO-Ziel)
      robust_std        Std derselben Stichprobe (Empfindlichkeit von x_rec)
      context_expected  E_{c~P*}[f(x_rec_steuer, c)]                       (nur Kontext-Szenario)
      context_worstcase inf_{MMD(P,P*)<=eps} E_{c~P}[f(x_rec_steuer, c)]   (DRBO-Ziel, nur Kontext-Szenario)

    Regrets werden NICHT hier gespeichert, sondern im Post-Processing aus
    f_opt und diesen Werten berechnet (siehe Post_Processing/derived_metrics.py).
    """
    x_rec = np.asarray(x_rec, dtype=float).reshape(-1)
    _, gauss, contexts = _metric_sets(len(x_rec), lower_bound, upper_bound)

    expected_vals = _evaluate_many(f, np.clip(x_rec + gauss, lower_bound, upper_bound))
    metrics = {
        "f_true_rec": f(x_rec),
        "robust_worstcase": _robust_worstcase(f, x_rec, lower_bound, upper_bound),
        "robust_expected": float(expected_vals.mean()),
        "robust_std": float(expected_vals.std(ddof=1)),
        "context_expected": float("nan"),
        "context_worstcase": float("nan"),
    }

    if config.CONTEXT_SCENARIO_ENABLED:
        points = np.column_stack([np.tile(x_rec[:-1], (len(contexts), 1)), contexts])
        context_vals = _evaluate_many(f, points)
        metrics["context_expected"] = float(context_vals.mean())
        metrics["context_worstcase"] = _context_worstcase(context_vals, contexts)

    return metrics


def select_and_run_variant(surrogate_model, acquisition_func, train_X, train_Y, dim,
                            lower_bound, upper_bound):# ✅
    """Analog zu waehle_und_berechne_variante() in ROBO/coco_benchmark.py.

    Alle Varianten haben dieselbe Signatur (train_X, train_Y, dim, acquisition_func,
    lower_bound, upper_bound) und geben (next_x, acq_value, extras) zurueck.
    extras ist ein dict mit mindestens "posterior_mean" (Funktion X_np ->
    Posterior-Mittelwert, fuer den empfohlenen Punkt x_rec), bei STABLEOPT
    zusaetzlich "x_tilde"/"x_tilde_min_lcb" (Empfehlungsregel des Papers). Siehe die
    jeweiligen run_*()-Funktionen in gaussian_process.py / imprecise_gp.py /
    relevance_pursuit.py / stable_opt.py / drbo.py / airbo.py fuer Details und
    modellspezifische Einschraenkungen (Imprecise GP: nur dim=1; STABLEOPT: nur
    acquisition_func="UCB"; DRBO: nur acquisition_func="distributionally
    robust UCB-Akquisition", nutzt die letzte Spalte von dim als
    Kontextvariable; AIRBO: nur acquisition_func="UCB", nimmt eine simulierte
    Gauss-Eingabeunsicherheit um jeden Punkt an, siehe airbo.py).
    """
    if surrogate_model == "GausianProzess":
        from gaussian_process import run_gaussian_process
        return run_gaussian_process(train_X, train_Y, dim, acquisition_func, lower_bound, upper_bound)
    elif surrogate_model == "Imprecise_GausianProzess_Rodemann":
        from imprecise_gp import run_imprecise_gausian_prozess
        return run_imprecise_gausian_prozess(train_X, train_Y, dim, acquisition_func, lower_bound, upper_bound)
    elif surrogate_model == "AIRBO":
        from airbo import run_airbo
        return run_airbo(train_X, train_Y, dim, acquisition_func, lower_bound, upper_bound)
    elif surrogate_model == "STABLEOPT":
        from stable_opt import run_stable_opt
        return run_stable_opt(train_X, train_Y, dim, acquisition_func, lower_bound, upper_bound)
    elif surrogate_model == "DRBO":
        from drbo import run_drbo
        return run_drbo(train_X, train_Y, dim, acquisition_func, lower_bound, upper_bound)
    elif surrogate_model == "Relevance Pursuit":
        # eigener Import-Block, siehe Hinweis oben: schlaegt mit aelterem botorch
        # gezielt erst HIER fehl, statt schon beim Modul-Import von core_bo.py
        try:
            from relevance_pursuit import run_relevance_pursuit
        except ImportError as e:
            raise ImportError(
                "Variante 'Relevance Pursuit' braucht RobustRelevancePursuitSingleTaskGP "
                "(botorch.models.robust_relevance_pursuit_model bzw. bei aelteren Versionen "
                "botorch.models.relevance_pursuit_model) - das gibt es erst ab botorch >= 0.11, "
                "was wiederum Python >= 3.10 voraussetzt. Mit der hier installierten botorch-/"
                "Python-Version nicht verfuegbar - siehe RoBo_Benchmark/.venv311 (Python 3.11 + "
                "aktuelles botorch) fuer eine Umgebung, in der diese Variante laeuft."
            ) from e
        return run_relevance_pursuit(train_X, train_Y, dim, acquisition_func, lower_bound, upper_bound)
    else:
        raise ValueError(f"Unbekanntes Surrogatmodell: {surrogate_model}")


def _task_seed(base_seed, task):
    """Deterministischer Seed aus (funktion, dimension, tensor_block) - bewusst
    OHNE combination_id/surrogate_model: alle Verfahren bekommen auf demselben
    (Funktion, Dimension, Tensorblock) exakt dieselbe Rauschfolge (Common Random
    Numbers -> fairer Vergleich mit weniger Varianz), unabhaengig davon, welche
    Paare in config.SURROGATE_AKQUISITION_PAARE aktiv sind. Ein Rerun (z.B. nach
    Abbruch) bekommt ebenfalls dasselbe Rauschen. Rueckgabe als Liste fuer
    np.random.default_rng (intern SeedSequence -> kollisionsfrei)."""
    return [int(base_seed), int(task["funktion"]), int(task["dimension"]), int(task["tensor_block"])]


def _seed_global_rngs(task):
    """Setzt die globalen RNGs von torch und numpy deterministisch pro Task
    (siehe config.BO_SEED). Wird zu Beginn jeder Task aufgerufen - pro Task
    und nicht einmal pro Worker-Prozess, damit das Ergebnis unabhaengig davon
    ist, welcher Worker die Task in welcher Reihenfolge abarbeitet."""
    seed = int(np.random.SeedSequence(_task_seed(config.BO_SEED, task)).generate_state(1)[0])
    torch.manual_seed(seed)
    np.random.seed(seed)


def _make_y_noise_fn(task):# ✅
    """Baut eine Funktion add_noise(y, aktiv), die -f(x)-Werte optional mit
    Rauschen versieht. aktiv ist config.NOISE_Y_INIT_ENABLED fuer die initialen
    train_Y bzw. config.NOISE_Y_ITER_ENABLED fuer die BO-Iterationen. Ein
    gemeinsamer RNG fuer beide Phasen, siehe _task_seed().
    """
    rng = np.random.default_rng(_task_seed(config.NOISE_Y_SEED, task))

    def add_noise(y, aktiv):
        if aktiv and rng.random() < config.NOISE_Y_ANTEIL:
            return y + rng.normal(loc=0.0, scale=config.NOISE_Y_SIGMA)
        return y

    return add_noise


def _make_x_iter_noise_fn(task, lower_bound, upper_bound):
    """Baut eine Funktion, die das vorgeschlagene next_x in die tatsaechlich
    ausgewertete Position next_x + eps uebersetzt (Eingaberauschen pro
    Iteration). Nur aktiv, wenn config.NOISE_X_ITER_ENABLED True ist; das
    Ergebnis wird auf [lower_bound, upper_bound] geclippt. Seed siehe
    _task_seed().
    """
    if not config.NOISE_X_ITER_ENABLED:
        return lambda x: x

    rng = np.random.default_rng(_task_seed(config.NOISE_X_ITER_SEED, task))

    def add_noise(x):
        if rng.random() < config.NOISE_X_ITER_ANTEIL:
            eps = rng.normal(loc=0.0, scale=config.NOISE_X_ITER_SIGMA, size=x.shape)
            return np.clip(x + eps, lower_bound, upper_bound)
        return x

    return add_noise


def run_task(task, suite, tensor_noise, bounds, progress_callback=None):# ✅ 
    """Fuehrt EINE Task komplett aus: ein (Kombination, Tensorblock)-Paar.

    Gibt eine Liste von Zeilen zurueck, eine pro BO-Iteration (Long-Format),
    analog zu den Var_n-Spalten in df_gesamt_iteration im alten Skript.

    progress_callback: optional, wird nach jeder abgeschlossenen Iteration mit
    der aktuellen Iterationsnummer j aufgerufen (fuer Live-Fortschrittsanzeige,
    siehe run_instance.py). Standardmaessig None (kein Overhead, z.B. wenn
    run_task ausserhalb der Pipeline direkt aufgerufen wird).
    """
    funktion = int(task["funktion"])
    dimension = int(task["dimension"])
    surrogate_model = task["surrogate_model"]
    acquisition_model = task["acquisition_model"]
    sample_size_initial = int(task["sample_size_initial"])
    max_iteration = int(task["max_iteration"])
    tensor_block = int(task["tensor_block"])

    lower_bound, upper_bound = bounds

    if config.CONTEXT_SCENARIO_ENABLED and dimension < 2:
        raise ValueError("Kontext-Szenario braucht dimension >= 2 (letzte Koordinate = Kontext).")

    # bereits in prepare_tasks.py einmalig pro (Funktion, Dimension) geschaetzt
    f_opt = float(task["f_opt_approx"])
    x_opt = np.asarray(json.loads(task["x_opt_approx"]), dtype=float)

    _seed_global_rngs(task)
    add_y_noise = _make_y_noise_fn(task)
    add_x_iter_noise = _make_x_iter_noise_fn(task, lower_bound, upper_bound)
    # eigener RNG fuer die Umgebungs-Kontexte (Common Random Numbers ueber Verfahren)
    context_rng = np.random.default_rng(_task_seed(config.CONTEXT_SEED, task))

    # Problem EINMAL pro Task holen (statt pro Auswertung) und am Ende freigeben.
    # f ist rauschfrei und negiert (Maximierung), wie evaluate_bbob_function.
    problem = suite.get_problem_by_function_dimension_instance(
        function=funktion, dimension=dimension, instance=1
    )

    def f(x):
        return -float(problem(np.asarray(x, dtype=float).reshape(-1)))

    try:
        block = tensor_noise[tensor_block]
        train_X_np = np.array(block[:sample_size_initial, :dimension], dtype=float)
        if config.CONTEXT_SCENARIO_ENABLED:
            # auch die initialen Kontexte kommen aus der Umgebung (c ~ P*)
            train_X_np[:, -1] = _sample_context(context_rng, sample_size_initial, lower_bound, upper_bound)

        f_true_history = [f(x) for x in train_X_np]
        train_Y_np = np.array([add_y_noise(v, config.NOISE_Y_INIT_ENABLED) for v in f_true_history])

        train_X = torch.as_tensor(train_X_np, dtype=torch.double)
        train_Y = torch.as_tensor(train_Y_np, dtype=torch.double).unsqueeze(-1)

        # StableOpt-Empfehlung nach dem Paper: x~_t* mit t* = argmax_t min_delta lcb_{t-1}(x~_t + delta)
        x_tilde_star, x_tilde_star_lcb = None, -np.inf

        rows = []
        for j in range(1, max_iteration + 1):
            next_x, _, extras = select_and_run_variant(
                surrogate_model, acquisition_model, train_X, train_Y, dimension,
                lower_bound, upper_bound,
            )
            next_x_np = next_x.detach().cpu().numpy().astype(float).copy()
            if config.CONTEXT_SCENARIO_ENABLED:
                # Kontext gibt die Umgebung vor, nicht das Verfahren (Kirschner et al. 2020)
                next_x_np[-1] = _sample_context(context_rng, 1, lower_bound, upper_bound)[0]

            # Modell/train_X sieht next_x, ausgewertet wird ggf. die verrauschte Position
            eval_x_np = add_x_iter_noise(next_x_np)
            if config.CONTEXT_SCENARIO_ENABLED:
                eval_x_np = np.array(eval_x_np, dtype=float)
                eval_x_np[-1] = next_x_np[-1]  # Kontext wird exakt beobachtet, kein Eingaberauschen
            f_true_xt = f(eval_x_np)
            next_y_np = add_y_noise(f_true_xt, config.NOISE_Y_ITER_ENABLED)

            # Empfohlener Punkt x_rec: unter den Punkten, auf denen das Modell in
            # dieser Iteration gefittet wurde, der mit dem hoechsten Posterior-
            # Mittelwert des Verfahrens. next_x selbst ist bewusst NICHT Kandidat:
            # das Modell kennt dessen y noch nicht und wuerde dort nur extrapolieren
            # (bei unskalierten Modellen wie AIRBO/Imprecise GP faellt der
            # Mittelwert dort auf den Prior-Mittelwert 0 zurueck). next_x wird ab
            # der naechsten Iteration Kandidat (eine Iteration Verzug).
            posterior_mean = extras.get("posterior_mean")
            if posterior_mean is not None:
                x_rec = train_X_np[int(np.argmax(posterior_mean(train_X_np)))]
            else:
                x_rec = train_X_np[int(np.argmax(train_Y_np))]
            candidates = np.vstack([train_X_np, next_x_np.reshape(1, -1)])

            if "x_tilde" in extras and extras["x_tilde_min_lcb"] > x_tilde_star_lcb:
                x_tilde_star, x_tilde_star_lcb = extras["x_tilde"], extras["x_tilde_min_lcb"]

            train_X_np = candidates
            train_Y_np = np.append(train_Y_np, next_y_np)
            f_true_history.append(f_true_xt)
            train_X = torch.as_tensor(train_X_np, dtype=torch.double)
            train_Y = torch.as_tensor(train_Y_np, dtype=torch.double).unsqueeze(-1)

            best_so_far, simple_regret, immediate_regret, x_distances = compute_regret(
                train_X_np, train_Y_np, x_opt, f_opt
            )
            metrics = compute_all_metrics(f, x_rec, lower_bound, upper_bound)

            rows.append({
                **task,
                "iteration": j,
                # verrauschte Kennzahlen (wie bisher, zur Vergleichbarkeit mit alten Laeufen)
                "best_so_far": float(best_so_far[-1]),
                "simple_regret": float(simple_regret[-1]),
                "immediate_regret": float(immediate_regret[-1]),
                "x_distance": float(x_distances[-1]),
                # rauschfreie Kennzahlen (siehe compute_all_metrics)
                "f_true_xt": f_true_xt,
                "best_true_so_far": float(np.max(f_true_history)),
                **metrics,
                "stableopt_worstcase_xtilde": (
                    _robust_worstcase(f, x_tilde_star, lower_bound, upper_bound)
                    if x_tilde_star is not None else float("nan")
                ),
                "status": "ok",
                "error": "",
            })

            if progress_callback is not None:
                progress_callback(j)
    finally:
        problem.free()

    return rows
