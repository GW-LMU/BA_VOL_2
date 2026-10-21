# preprocessing/seed_manager.py

import hashlib
import numpy as np

from preprocessing.config import MASTER_SEED, N_WIEDERHOLUNG


def make_seed(purpose, repetition):

    key = f"{MASTER_SEED}|{purpose}|{repetition}"

    digest = hashlib.sha256(
        key.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little")





# Erstellt einen unabhaengigen NumPy-Generator
def get_rng(purpose, repetition):
    return np.random.default_rng(make_seed(purpose, repetition))





#Liefert die Seeds aller Wiederholungen fuer einen Zweck.
def get_seeds(purpose):
    return [make_seed(purpose, i) for i in range(N_WIEDERHOLUNG)]





# Feste Seed-Listen pro Zweck, Index = Wiederholung
START_SAMPLE_SEEDS = get_seeds("initial_design")
PRIOR_SEEDS = get_seeds("prior")
INPUT_NOISE_SEEDS = get_seeds("input_noise")
OUTPUT_NOISE_SEEDS = get_seeds("output_noise")
ADVERSARIAL_SEEDS = get_seeds("adversarial")
CONTEXT_SEEDS = get_seeds("context")
ALGORITHM_SEEDS = get_seeds("algorithm")
EVALUATION_SEEDS = get_seeds("evaluation")
