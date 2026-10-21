import numpy as np
import torch

from main_processing.scenarios.nominal import build_tensor   # gemeinsames Startdesign
from preprocessing import config
from preprocessing import seed_manager as sdm


# Szenario "Prior-Fehler" nach Rodemann & Augustin (2024), Imprecise Bayesian
# Optimization, Knowledge-Based Systems 300, 112186.
#
# Die Trainingsdaten bleiben UNVERAENDERT. Gestoert wird das Vorwissen des
# Surrogatmodells: Der konstante Prior-Mittelwert m(x) = c des GP wird bewusst
# falsch gesetzt und beim Fitten eingefroren (kein Empirical Bayes).
#
# Die Intensitaet (config.PRIOR_SIGMAS) ist der Abstand von c zum empirischen
# Mittelwert der Startstichprobe, gemessen in deren Standardabweichungen
# (das Surrogat arbeitet mit Standardize, daher entspricht 1.0 genau 1 Std.).
#
# Richtung (BoTorch maximiert, das Surrogat sieht y = -f):
#   +1 = optimistisch  -> Modell erwartet in unbekannten Regionen gute Werte
#   -1 = pessimistisch -> Modell erwartet in unbekannten Regionen schlechte Werte


def build_prior_offsets(intensity, direction="random"):
    """
    Erzeugt pro Wiederholung den Fehler des Prior-Mittelwerts
    (in standardisierten y-Einheiten).

    direction:
        "random"      -> Vorzeichen pro Wiederholung zufaellig (+/-), mit eigenem Seed
        "optimistic"  -> immer +intensity
        "pessimistic" -> immer -intensity
    """

    offsets = np.empty(config.N_WIEDERHOLUNG, dtype=np.float64)

    for i in range(config.N_WIEDERHOLUNG):
        if direction == "random":
            rng = sdm.get_rng("prior", i)  # eigener Seed pro Wiederholung
            sign = rng.choice([-1.0, 1.0])
        elif direction == "optimistic":
            sign = 1.0
        elif direction == "pessimistic":
            sign = -1.0
        else:
            raise ValueError(f"Unbekannte Prior-Richtung: {direction}")

        offsets[i] = sign * intensity

    return offsets


def evaluate_prior_error_function(next_x, intensity, objective, rng):
    return objective(next_x)                    # Daten ungestoert, nur der Prior ist falsch

def run_prior_error_function(scenario, intensity, direction="random"):
    tensor = build_tensor()                                 # saubere Startpunkte, identisch zu nominal
    prior_offsets = build_prior_offsets(intensity, direction)  # ein Wert pro Wiederholung

    return tensor, prior_offsets          # 2 Werte, wie bei nominal

def apply_prior_error(model, prior_offset):
    """
    Setzt den konstanten Prior-Mittelwert eines BoTorch-GP (z. B. SingleTaskGP
    mit Standardize) auf prior_offset und friert ihn ein.
    prior_offset ist EIN float (der Wert der aktuellen Wiederholung,
    also prior_offsets[repetition] - indiziert wird im experiment_runner).
    MUSS vor fit_gpytorch_mll(...) aufgerufen werden, sonst schaetzt der Fit
    den Fehler sofort wieder weg.
    """

    mean_module = model.mean_module

    with torch.no_grad():
        mean_module.raw_constant.fill_(float(prior_offset))
    mean_module.raw_constant.requires_grad_(False)

    return model