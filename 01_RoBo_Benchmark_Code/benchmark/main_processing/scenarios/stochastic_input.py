import numpy as np

from main_processing.scenarios.nominal import build_tensor   # gemeinsames Startdesign


# Szenario "stochastischer Input": Der Optimierer waehlt x, ausgewertet wird
# aber an x + xi mit xi ~ N(0, sigma^2 I) - bei JEDER Auswertung (Startpunkte
# und Iterationen) mit frischem xi. Gespeichert wird das unverrauschte x,
# d. h. das Modell sieht (x, f(x + xi)).
# rng kommt pro Wiederholung aus dem experiment_runner
# (sdm.get_rng("input_noise", repetition)) -> Auswertung k bekommt bei allen
# Algorithmen dasselbe xi (Common Random Numbers).

def evaluate_stochastic_input_function(next_x, intensity, objective, rng):
    x = np.asarray(next_x, dtype=float).reshape(-1)
    xi = rng.normal(loc=0.0, scale=intensity, size=x.shape)
    return objective(x + xi)


def run_stochastic_input_function(scenario, intensity):
    tensor = build_tensor()               # saubere Startpunkte, identisch zu nominal
    prior_offsets = None                  # kein Prior-Fehler
    return tensor, prior_offsets
