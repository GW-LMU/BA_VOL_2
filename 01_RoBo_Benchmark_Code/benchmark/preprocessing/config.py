
from pathlib import Path

# Define the base directory and data directory for the benchmark code
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"




ALGORITHM_CONFIGS = {

    "GP_BO": {
        "surrogate_model": "GP_BO",
        "acquisition": "EI",
        "kernel": "Matern52",
    },

    "PROBO": {
        "surrogate_model": "PROBO",
        "acquisition": "GLCB",
        "kernel": "Matern32",           # uncertainty_aware_bo.matern ist Matern-3/2
        "prior_set_width": 0.5,
    },

    "RP_BO": {
            "surrogate_model": "RP_BO",
            "acquisition": "EI",                 # wie GP_BO, nur Ausreisser-Likelihood anders
            "kernel": "Matern52",
            "beta": 2.0,
        },

    "STABLEOPT": {  
        "surrogate_model": "STABLEOPT",
        "acquisition": "UCB",                # robuste Min-Max-UCB (Bogunovic et al. 2018)
        "kernel": "Matern52",
        "beta": 2.0,
    },

    "DRBO": {
        "surrogate_model": "DRBO",
        "acquisition": "UCB",                # distributionally robust UCB (Kirschner et al. 2020)
        "kernel": "Matern52",
        "ambiguity_metric": "MMD",
        "mmd_kernel": "RBF",
    },

    "AIRBO": {
        "surrogate_model": "AIRBO",
        "acquisition": "UCB",                # UCB auf dem MMD-GP (Yang et al. 2023)
        "kernel": "MMD",
        "nystrom_rank": 100,
    },
}


# Verfahren, die tatsaechlich gerechnet werden (muessen in ALGORITHM_CONFIGS stehen).
# RP_BO, STABLEOPT, DRBO, AIRBO sind an die Pipeline angepasst, aber noch nicht aktiviert.
ALGORITHMS = ["GP_BO", 
              "PROBO", 
              #"RP_BO", 
              #"STABLEOPT", 
              #"DRBO", 
              #"AIRBO"
              ]


FUNCTIONS = [1]
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  
        #   [1,6,11,16,21]  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
DIMENSIONS = [2, 
              5,
              10
              ]   
                                                                                               #    
INSTANCES = [1]                                                                                              # Bleibt bei 1, da die Benchmark-Funktionen nur eine Instanz haben

N_START_SAMPLES = 5
N_WIEDERHOLUNG = 10
EVALUATION_BUDGET = 100                                   # Auswertungen gesamt, inkl. Startpunkte
N_ITERATIONS = EVALUATION_BUDGET - N_START_SAMPLES        # BO-Iterationen pro Lauf
MASTER_SEED = 42

# Parallelisierung: Experimente sind unabhaengig und werden auf mehrere Prozesse verteilt.
#   N_JOBS = 1  -> sequentiell (einfacher zum Debuggen)
#   N_JOBS = 6  -> 6 Experimente gleichzeitig
#   N_JOBS = -1 -> alle Kerne,  -2 -> alle bis auf einen, usw.

N_JOBS = 10
THREADS_PER_JOB = 1                 # Torch/BLAS-Threads pro Prozess (1 vermeidet Ueberbelegung)

PRIOR_SIGMAS = [0.0, 
                0.5, 
                1.0
                ]




INPUT_SIGMAS = [0.0, 
                0.25,
                0.5
                ]



#ADVERSARIAL_EPSILONS = [0.0, 0.1, 0.3, 0.5]
#SHIFT_MAGNITUDES = [0.0, 0.1, 0.3, 0.5]


SCENARIOS = ["nominal", 
             "prior_error", 
             "stochastic_input", 
             #"adversarial_input", 
             #"distribution_shift"
             ]


BOUNDS = [-5.0, 5.0]
NOISE_ANTEIL = 1.0


# Parameter der Robustheitsmetriken (postprocessing/metrics.py).
# Bewusst FEST und fuer alle Szenarien gleich, damit die Metriken vergleichbar sind.
METRIC_SIGMA = 0.25                 # Std. des Eingaberauschens (M2, M4, M5, M6)
METRIC_EPSILON = 0.25               # Radius des Worst-Case-Wuerfels (M3)
METRIC_SHIFT = 0.3                  # Verschiebung des Rausch-Mittelwerts (M4)
METRIC_DRO_RHO = 0.1                # KL-Radius der Ambiguitaetsmenge (M5)
METRIC_CVAR_ALPHA = 0.9             # CVaR-Niveau (M6)
METRIC_N_MC = 500                   # Monte-Carlo-Stichproben
METRIC_N_WORST_CASE = 500           # Kandidaten der Worst-Case-Suche
METRIC_N_RESTARTS = 3               # lokale Verfeinerungen der Worst-Case-Suche
METRIC_N_SHIFT_DIRECTIONS = 8       # Richtungen fuer den Shift