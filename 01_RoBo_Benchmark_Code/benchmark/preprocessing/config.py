
from pathlib import Path

# Define the base directory and data directory for the benchmark code
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"




# Einstellungen je Verfahren. "surrogate_model" und "acquisition" waehlen das Verfahren,
# alle uebrigen Eintraege werden vom Runner als Keyword-Argumente an die run_*()-Funktion
# des Verfahrens uebergeben (siehe experiment_runner.select_and_run_variant).
# Kernel der BoTorch-Verfahren (GP_BO, RP_BO, STABLEOPT, DRBO): "RBF_ARD", "Matern52_ARD"
# oder "Matern32_ARD", jeweils mit dimensionsskaliertem LogNormal-Prior (Hvarfner et al. 2024).
ALGORITHM_CONFIGS = {

    "GP_BO": {
        "surrogate_model": "GP_BO",
        "acquisition": "LCB",             # "EI" oder "LCB" (Minimierung)
        "kernel": "RBF_ARD",
        "beta": 2.0,                    # nur fuer LCB
        "num_restarts": 10,             # optimize_acqf
        "raw_samples": 512,
    },

    "PROBO": {
        "surrogate_model": "PROBO",
        "acquisition": "GLCB",
        "kernel": "Matern32",           # "Matern32" oder "RBF" (uncertainty_aware_bo)
        # Imprecision-Parameter. c = 1 statt 50 (Default in uncertainty_aware_bo): Auf standardisierten
        # y ist die Credal-Set-Breite 2c/S_k * |w(x)| dann in derselben Groessenordnung
        # wie Mittelwert- und Varianzterm. Bei c = 50 dominiert sie um Faktor 50-100,
        # GLCB wird fast reine Exploration und ueberdeckt einen Prior-Fehler delta.
        "c": 1.0,
        # True: length_scale/outputscale werden in jeder Iteration per Marginal
        # Likelihood gefittet, wie bei GP_BO/DRBO/StableOpt/RP_BO (im Szenario
        # prior_error unter dem eingefrorenen falschen Prior-Mittelwert).
        # False: die festen Werte unten werden verwendet (alte Version).
        "fit_hyperparameters": True,
        "length_scale": 1.0,            # nur bei fit_hyperparameters = False
        "outputscale": 1.0,             # nur bei fit_hyperparameters = False
        "prior_mean_max": 5.0,          # Prior-Mittelwert-Menge M in [-5, 5]
        "n_prior_means": 250,           # Anzahl M-Kandidaten je Vorzeichen
        "tau": 1.0,                     # Gewicht des Varianzterms
        "rho": 1.0,                     # Gewicht der Credal-Set-Breite (nur GLCB)
        "focus_points": 1000,           # Focus Search (Rodemann & Augustin 2024)
        "focus_maxit": 5,
        "focus_restarts": 5,
    },

    "RP_BO": {
        "surrogate_model": "RP_BO",
        "acquisition": "LCB",            # wie GP_BO, nur Ausreisser-Likelihood anders
        "kernel": "RBF_ARD",
        "beta": 2.0,                    # nur fuer LCB
        "num_restarts": 10,
        "raw_samples": 512,
    },

    "STABLEOPT": {
        "surrogate_model": "STABLEOPT",
        "acquisition": "LCB",           # robuste Min-Max-LCB (Bogunovic et al. 2018, Minimierungs-Form)
        "kernel": "RBF_ARD",
        "beta": 2.0,
        "epsilon": None,                # Stabilitaetsradius; None = METRIC_EPSILON (wie M3)
        "norm": "linf",                 # "linf" = eps-Wuerfel (wie M3), "l2" = eps-Kugel (Paper-Beispiel)
        "n_delta_candidates": 64,       # Zufallspunkte der Stoerungsmenge (+ delta=0 und 2*dim Achsenpunkte)
        "num_restarts": 10,             # optimize_acqf, wie GP_BO
        "raw_samples": 512,
    },

    "DRBO": {
        "surrogate_model": "DRBO",
        "acquisition": "LCB",           # distributionally robust LCB (Kirschner et al. 2020, Minimierungs-Form), MMD mit RBF-Kernel
        "kernel": "RBF_ARD",
        "beta": 2.0,
        "context_dim": 1,               # letzte Spalten von x = Kontext
        "n_context_candidates": 15,
        "n_x_candidates": 200,
        "mmd_lengthscale": 1.0,
        "delta": 0.1,                   # Konfidenz fuer den MMD-Radius (Lemma 3)
    },

    "AIRBO": {
        "surrogate_model": "AIRBO",
        "acquisition": "LCB",           # LCB auf dem MMD-GP (Yang et al. 2023, Minimierungs-Form)
        "beta": 2.0,
        "noise_std": None,              # Eingabeunsicherheit; None = 1% der Suchraumbreite
        "n_cloud_samples": 50,
        "use_nystrom": True,
        "sub_samp_size": 20,            # Nystroem-Rang h
        "mmd_length_scale": 1.0,
        "mmd_alpha": 1.0,
        "noise_var": 1e-6,
        "n_x_candidates": 200,
    },
}


# Verfahren, die tatsaechlich gerechnet werden (muessen in ALGORITHM_CONFIGS stehen).
# RP_BO, STABLEOPT, DRBO, AIRBO sind an die Pipeline angepasst, aber noch nicht aktiviert.
ALGORITHMS = ["GP_BO", 
              "PROBO", 
              #"RP_BO", 
              "STABLEOPT", 
              #"DRBO", 
              #"AIRBO"
              ]


FUNCTIONS = [16,20,24]
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  
        #   [1,6,11,16,21]  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
DIMENSIONS = [2, 
              5,
              10
              ]   
                                                                                               #    
INSTANCES = [1]                                                                                              # Bleibt bei 1, da die Benchmark-Funktionen nur eine Instanz haben

N_START_SAMPLES = 10
N_WIEDERHOLUNG = 30
EVALUATION_BUDGET = 100                                # Auswertungen gesamt, inkl. Startpunkte
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