from functools import lru_cache

import numpy as np
from scipy.stats import qmc

from preprocessing import seed_manager as sdm
from preprocessing import config


def _latin_hypercube(d, seed):
    # scipy >= 1.15 nimmt rng=, aeltere Versionen seed= (gleicher Zufallsstrom)
    generator = np.random.default_rng(seed)
    try:
        return qmc.LatinHypercube(d=d, scramble=True, rng=generator)
    except TypeError:
        return qmc.LatinHypercube(d=d, scramble=True, seed=generator)


# Baut eine Tensor mit LHS wie im Paper von Rodemann & Augustin (2024), Imprecise Bayesian Optimization, Knowledge-Based Systems 300, 112186.
# Gemeinsames Startdesign fuer ALLE Szenarien (Common Random Numbers).
# Gecacht, da es nicht vom Szenario abhaengt -> nur einmal pro Prozess gebaut.
@lru_cache(maxsize=1)
def _build_tensor_cached():
    max_dim = max(config.DIMENSIONS)
    max_samples = config.N_START_SAMPLES
    lower, upper = config.BOUNDS

    tensor = np.empty(
        (config.N_WIEDERHOLUNG, max_samples, max_dim),   # Tiefe: N_WIEDERHOLUNG, Zeilen: max_samples, Spalten: max_dim
        dtype=np.float64)

    for i in range(config.N_WIEDERHOLUNG):
        sampler = _latin_hypercube(max_dim, sdm.START_SAMPLE_SEEDS[i])

        # Latin Hypercube Sampling in [0, 1]
        samples = sampler.random(n=max_samples)

        # Skalierung auf [lower, upper]
        tensor[i] = qmc.scale(
            samples,
            l_bounds=np.full(max_dim, lower),
            u_bounds=np.full(max_dim, upper)
        )

    tensor.flags.writeable = False       # Schutz: Cache darf nicht veraendert werden
    return tensor


def build_tensor():
    return _build_tensor_cached()


def evaluate_nominal_function(next_x, intensity, objective, rng):
    return objective(next_x)


def run_nominal_function(scenario, intensity):
    tensor = build_tensor()
    prior_offsets = None                  # kein Prior-Fehler
    return tensor, prior_offsets
