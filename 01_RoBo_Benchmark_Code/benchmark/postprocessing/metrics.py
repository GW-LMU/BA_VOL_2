# postprocessing/metrics.py
#
# Robustheitsmetriken fuer den EMPFOHLENEN Punkt x eines BO-Laufs.
#
# Alle Metriken werden auf der WAHREN, rauschfreien BBOB-Funktion f berechnet
# (nicht auf den beobachteten next_y, die je nach Szenario gestoert sind).
# Gerechnet wird im Minimierungs-Sinn von COCO und immer als Regret relativ
# zu f_opt:  0 = optimal, groesser = schlechter. Damit sind alle Metriken
# ueber Funktionen und Szenarien hinweg vergleichbar.
#
# benchmark: Objekt mit
#     benchmark.f(X)    -> f-Werte fuer ein Array X der Form (n, d) (roh, Minimierung)
#     benchmark.f_opt   -> exakter Optimalwert f*
#     benchmark.bounds  -> (lower, upper)
#   (siehe BBOBBenchmark / make_benchmark unten)
#
# config: Modul/Objekt mit den METRIC_*-Parametern (preprocessing/config.py).
#
# Alle Stoerungen werden mit festen Seeds erzeugt (Common Random Numbers):
# Jeder Algorithmus wird mit exakt denselben Stoerungen bewertet.

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

from preprocessing import seed_manager as sdm


# ----------------------------------------------------------------------
# Benchmark-Wrapper
# ----------------------------------------------------------------------

class BBOBBenchmark:

    def __init__(self, problem, f_opt, bounds):
        self.problem = problem
        self.f_opt = float(f_opt)
        self.bounds = tuple(bounds)

    def f(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.array([self.problem(x) for x in X], dtype=float)


def make_benchmark(suite, function, dimension, true_max, bounds, instance=1):
    """
    true_max ist der negierte Optimalwert aus get_bbob_fopt_exact (-f_opt),
    daher f_opt = -true_max.
    Wichtig: Am Problem darf KEIN Observer haengen, sonst landen die
    Metrik-Auswertungen in den COCO-Logs.
    """

    problem = suite.get_problem_by_function_dimension_instance(
        function=function, dimension=dimension, instance=instance
    )
    return BBOBBenchmark(problem, f_opt=-true_max, bounds=bounds)


# ----------------------------------------------------------------------
# Hilfsfunktionen
# ----------------------------------------------------------------------

_SAMPLE_CACHE = {}


def _param(config, name, default):
    return getattr(config, name, default)


def _fixed_samples(purpose, n, d, kind="normal"):
    """Feste Zufallszahlen pro (Zweck, n, d) -> Common Random Numbers."""

    key = (purpose, n, d, kind)
    if key not in _SAMPLE_CACHE:
        rng = sdm.get_rng(purpose, 0)
        if kind == "normal":
            _SAMPLE_CACHE[key] = rng.standard_normal((n, d))
        else:
            _SAMPLE_CACHE[key] = rng.random((n, d))
    return _SAMPLE_CACHE[key]


def _as_point(x):
    return np.asarray(x, dtype=float).reshape(-1)


def _clip(X, benchmark):
    lower, upper = benchmark.bounds
    return np.clip(X, lower, upper)


def _input_noise_regrets(x, benchmark, config):
    """Regret-Stichprobe unter Eingaberauschen xi ~ N(0, sigma^2 I)."""

    x = _as_point(x)
    sigma = _param(config, "METRIC_SIGMA", 0.25)
    n_mc = _param(config, "METRIC_N_MC", 500)

    Z = _fixed_samples("metric_input_noise", n_mc, x.size)
    X = _clip(x + sigma * Z, benchmark)
    return benchmark.f(X) - benchmark.f_opt


def _cvar_from_regrets(regrets, alpha):
    # Mittelwert der schlechtesten (1 - alpha)-Anteile
    k = max(1, int(np.ceil((1.0 - alpha) * regrets.size)))
    return float(np.mean(np.sort(regrets)[-k:]))


def _kl_dro_from_regrets(regrets, rho):
    """
    sup_{Q : KL(Q||P) <= rho} E_Q[r]
      = min_{lambda > 0}  lambda * rho + lambda * log E_P[exp(r / lambda)]
    (Dualformulierung, Hu & Hong 2013). P = empirische Rausch-Verteilung.
    """

    if rho <= 0.0:
        return float(np.mean(regrets))

    n = regrets.size
    scale = max(float(np.std(regrets)), 1e-12)

    def dual(log_lambda):
        lam = np.exp(log_lambda)
        return lam * rho + lam * (logsumexp(regrets / lam) - np.log(n))

    res = minimize_scalar(
        dual,
        bounds=(np.log(scale) - 15.0, np.log(scale) + 15.0),
        method="bounded",
    )

    # Der Wert liegt theoretisch zwischen Mittelwert und Maximum
    return float(np.clip(res.fun, np.mean(regrets), np.max(regrets)))


# ----------------------------------------------------------------------
# Metriken
# ----------------------------------------------------------------------

def nominal_regret(x, benchmark):
    """M1: f(x) - f_opt ohne Stoerung."""

    return float(benchmark.f(_as_point(x))[0] - benchmark.f_opt)


def expected_performance(x, benchmark, config):
    """M2: E_xi[ f(x + xi) ] - f_opt mit xi ~ N(0, sigma^2 I) (Monte Carlo)."""

    return float(np.mean(_input_noise_regrets(x, benchmark, config)))


def worst_case_performance(x, benchmark, config):
    """
    M3: max_{||delta||_inf <= eps} f(x + delta) - f_opt   (StableOpt-Sicht).
    Approximiert durch Zufallssuche im eps-Wuerfel + lokale Verfeinerung
    (L-BFGS-B) der besten Kandidaten. Ergebnis ist eine untere Schranke
    des wahren Worst Case.
    """

    x = _as_point(x)
    eps = _param(config, "METRIC_EPSILON", 0.25)
    n_cand = _param(config, "METRIC_N_WORST_CASE", 500)
    n_restarts = _param(config, "METRIC_N_RESTARTS", 3)

    lower, upper = benchmark.bounds
    lo = np.maximum(x - eps, lower)
    hi = np.minimum(x + eps, upper)

    U = _fixed_samples("metric_worst_case", n_cand, x.size, kind="uniform")
    candidates = np.vstack([x, lo + U * (hi - lo)])
    values = benchmark.f(candidates)
    worst = float(np.max(values))

    box = list(zip(lo, hi))
    for start in candidates[np.argsort(values)[-n_restarts:]]:
        res = minimize(
            lambda z: -benchmark.f(z)[0],
            start,
            method="L-BFGS-B",
            bounds=box,
            options={"maxiter": 50},
        )
        worst = max(worst, float(-res.fun))

    return worst - benchmark.f_opt


def shift_performance(x, benchmark, config):
    """
    M4: Erwarteter Regret unter verschobener Eingabeverteilung
        xi ~ N(Delta * u, sigma^2 I), ||u|| = 1,
    gemittelt ueber K feste Zufallsrichtungen u.
    """

    x = _as_point(x)
    d = x.size
    sigma = _param(config, "METRIC_SIGMA", 0.25)
    delta = _param(config, "METRIC_SHIFT", 0.3)
    n_mc = _param(config, "METRIC_N_MC", 500)
    n_dir = _param(config, "METRIC_N_SHIFT_DIRECTIONS", 8)

    directions = _fixed_samples("metric_shift_direction", n_dir, d)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    Z = _fixed_samples("metric_input_noise", n_mc, d)

    means = []
    for u in directions:
        X = _clip(x + delta * u + sigma * Z, benchmark)
        means.append(np.mean(benchmark.f(X) - benchmark.f_opt))

    return float(np.mean(means))


def dro_performance(x, benchmark, config):
    """
    M5: Distributionally Robust Regret
        sup_{Q : KL(Q||P) <= rho} E_Q[ f(x + xi) ] - f_opt,
    P = Eingaberauschen N(0, sigma^2 I).
    """

    regrets = _input_noise_regrets(x, benchmark, config)
    rho = _param(config, "METRIC_DRO_RHO", 0.1)
    return _kl_dro_from_regrets(regrets, rho)


def cvar(x, benchmark, config):
    """M6: CVaR_alpha des Regrets unter Eingaberauschen (Mittel der schlechtesten 1-alpha)."""

    regrets = _input_noise_regrets(x, benchmark, config)
    alpha = _param(config, "METRIC_CVAR_ALPHA", 0.9)
    return _cvar_from_regrets(regrets, alpha)


# ----------------------------------------------------------------------
# Verlaufs-Metriken (bewerten den ganzen Lauf x_1 ... x_T, nicht einen Punkt)
# ----------------------------------------------------------------------

def instantaneous_regrets(y_history, benchmark, negated=False):
    """
    r_t = f(x_t) - f_opt fuer alle angefragten Punkte.

    y_history: Funktionswerte der angefragten Punkte (z. B. alle next_y).
    negated:   True, wenn y_history fuer BoTorch negiert ist (y = -f).

    Die y muessen die WAHREN f-Werte an x_t sein. Sind die beobachteten y
    gestoert (z. B. Ausgaberauschen), stattdessen benchmark.f(X_history) uebergeben.
    """

    y = np.asarray(y_history, dtype=float).reshape(-1)
    if negated:
        y = -y
    return y - benchmark.f_opt


def simple_regret(y_history, benchmark, negated=False):
    """
    Simple Regret je Iteration: S_t = min_{i <= t} r_i
    (Regret des besten bisher angefragten Punkts). S_T ist der finale Wert.
    """

    return np.minimum.accumulate(instantaneous_regrets(y_history, benchmark, negated))


def cumulative_regret(y_history, benchmark, negated=False):
    """
    Cumulative Regret je Iteration: R_t = sum_{i <= t} r_i.
    Misst, wie viel ein Algorithmus waehrend der Suche 'verschenkt'.
    """

    return np.cumsum(instantaneous_regrets(y_history, benchmark, negated))


def compute_all_metrics(x, benchmark, config):
    # Rausch-Stichprobe nur einmal auswerten und fuer M2, M5, M6 teilen
    regrets = _input_noise_regrets(x, benchmark, config)

    return {
        "nominal_regret":
            nominal_regret(x, benchmark),

        "expected_performance":
            float(np.mean(regrets)),

        "worst_case_performance":
            worst_case_performance(x, benchmark, config),

        "shift_performance":
            shift_performance(x, benchmark, config),

        "dro_performance":
            _kl_dro_from_regrets(regrets, _param(config, "METRIC_DRO_RHO", 0.1)),

        "cvar":
            _cvar_from_regrets(regrets, _param(config, "METRIC_CVAR_ALPHA", 0.9))
    }
