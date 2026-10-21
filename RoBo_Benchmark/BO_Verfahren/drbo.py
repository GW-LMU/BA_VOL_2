"""Distributionally-Robust-BO-Variante (Kirschner et al. 2020) fuer die verteilte Pipeline.

Angepasste Kopie von ROBO/DRBO.py: einziger Unterschied sind die
Default-Bounds, die jetzt aus config.py kommen (statt fest [-1000, 1000]).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pfade  # noqa: F401 - fuegt Pre_Processing/ (config) zu sys.path hinzu

import numpy as np
import scipy.optimize
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

import config
from gaussian_process import posterior_mean_fn


def _mmd_gram_matrix(context_candidates, mmd_lengthscale):
    """RBF-Kernel-Gram-Matrix kM(c_i, c_j) ueber die diskreten Kontext-Kandidaten (Kirschner et al. 2020, Sec. 2)."""
    diffs = context_candidates[:, None, :] - context_candidates[None, :, :]
    sqdist = np.sum(diffs ** 2, axis=-1)
    return np.exp(-0.5 * sqdist / mmd_lengthscale ** 2)

    # Hier Messung der Unsicherheit über die Kontextverteilung mittel Maxium Mean Dicrepnay Kernl 
    # Hier MMD +  Tranfomation in Kernl GP (Kirschner et al. 2020, Sec. 2)


def _solve_worst_case_weights(ucb_values, w_ref, M, epsilon):
    
    n = len(ucb_values)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: epsilon ** 2 - (w - w_ref) @ M @ (w - w_ref)},
    ]
    bounds = [(0.0, 1.0)] * n

    res = scipy.optimize.minimize(
        lambda w: w @ ucb_values,
        w_ref.copy(),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return res.x if res.success else w_ref


def run_drbo(train_X, train_Y, dim, acquisition_func,
             lower_bound=config.MINUS_AREA, upper_bound=config.PLUS_AREA,
             context_dim=1, n_context_candidates=15, n_x_candidates=200,
             mmd_lengthscale=1.0, delta=0.1, beta=2.0):
    """
    Fuehrt eine Iteration der Data-Driven-DRBO-Variante aus (Kirschner, Bogunovic, Jegelka,
    Krause 2020, "Distributionally Robust Bayesian Optimization", Algorithm 1 + Sec. 3.2):

    Die letzten `context_dim` Spalten von train_X werden als unkontrollierbarer Kontext c
    behandelt, die ersten (dim - context_dim) Spalten als Kontrollvariable x. Pro Runde:

    1. Empirische Kontextverteilung P_hat_t aus den bisher beobachteten Kontexten berechnen
    2. MMD-Unsicherheitsradius eps_t = (2 + sqrt(2*log(1/delta))) / sqrt(t-1)  (Lemma 3)
    3. Fuer jedes Kandidaten-x: w_ucb_x ueber das konvexe Programm bestimmen (worst-case
       Gewichtung der Kontext-Kandidaten unter dem UCB-Modell)
    4. x_t = argmax_x <w_ucb_x, ucb_x>
    5. naechster Kontext wird aus der worst-case-Verteilung w_ucb_{x_t} gezogen
       (Simulator-Setting-Approximation, da die Umgebung den echten Kontext vorgibt)

    train_X : torch.double, shape [n, dim]
    train_Y : torch.double, shape [n, 1] (Maximierungs-Konvention, siehe core_bo.py)
    dim     : int, Dimension des Gesamtraums (Kontrollvariablen + Kontextvariablen)

    Rueckgabe: next_x (torch.double, shape [dim]), acq_value (torch.double, Skalar),
               extras (dict, "posterior_mean": Funktion fuer core_bo.compute_all_metrics)
    """

    # Hier prüfen ob richtige Acqistionsfunktion gewählt worden ist
    if acquisition_func != "distributionally robust UCB-Akquisition":
        raise ValueError(
            f"DRBO verwendet intern eine feste, MMD-robuste UCB-Konstruktion; "
            f"unbekannte/inkompatible Akquisitionsfunktion: {acquisition_func}"
        )
    # Hier prüfen ob die dim  der Contextverteilung richtig ist 
    if context_dim >= dim:
        raise ValueError("context_dim muss kleiner als dim sein, damit ein Kontrollraum X uebrig bleibt.")

    control_dim = dim - context_dim

    bounds = torch.tensor(
        [[lower_bound] * dim, [upper_bound] * dim], dtype=torch.double
    )


    # Modell fitten (Inputs auf [0,1]^d normalisiert, Outputs standardisiert)
    model = SingleTaskGP(
        train_X, train_Y,
        input_transform=Normalize(d=dim, bounds=bounds),
        outcome_transform=Standardize(m=1),
    )

    
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Hier fitte ich das Modell damit ich die  Define (ucbtx)j := ˆft(x,cj) + βtσt(x,cj) bekomme
    # Fitten über RKHS Regression 

    # aus dem globalen numpy-RNG abgeleitet, der in core_bo.run_task() pro Task
    # geseedet wird (config.BO_SEED) -> reproduzierbar
    rng = np.random.default_rng(np.random.randint(2**31 - 1))

    # 1) empirische Kontextverteilung P_hat_t aus bisher beobachteten Kontexten
    observed_contexts = train_X[:, control_dim:].detach().cpu().numpy()
    n_obs = observed_contexts.shape[0]


    # Hier werden Anzahl an KOntroll_fim genutzt für x und eine für die Konetverteilung 

    grid_contexts = rng.uniform(lower_bound, upper_bound, size=(n_context_candidates, context_dim))
    context_candidates = np.vstack([observed_contexts, grid_contexts])

    w_ref = np.zeros(context_candidates.shape[0])
    for obs_c in observed_contexts:
        idx = np.argmin(np.sum((context_candidates - obs_c) ** 2, axis=1))
        w_ref[idx] += 1.0
    w_ref /= w_ref.sum()

    # 2) MMD-Unsicherheitsradius nach Lemma 3
    epsilon_t = (2.0 + np.sqrt(2.0 * np.log(1.0 / delta))) / np.sqrt(max(n_obs, 1))

    # Im sskript Lemma 3 - Seite 5/6 hier findet die wahl des optimalen Eplsion statt 

    # MMD-Kernel-Gram-Matrix ueber die Kontext-Kandidaten
    M = _mmd_gram_matrix(context_candidates, mmd_lengthscale)

    # 3)+4) fuer jedes Kandidaten-x den worst-case-gewichteten UCB-Wert bestimmen
    x_candidates = rng.uniform(lower_bound, upper_bound, size=(n_x_candidates, control_dim))
    n_c = context_candidates.shape[0]

    best_val = -np.inf
    best_x = None
    best_w = None

    with torch.no_grad():
        for x in x_candidates:
            xc_pairs = np.hstack([np.tile(x, (n_c, 1)), context_candidates])
            xc_tensor = torch.as_tensor(xc_pairs, dtype=torch.double)
            posterior = model.posterior(xc_tensor)
            mean = posterior.mean.squeeze(-1).numpy()
            std = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1).numpy()
            ucb = mean + (beta ** 0.5) * std

            w_worst = _solve_worst_case_weights(ucb, w_ref, M, epsilon_t)
            value = w_worst @ ucb

            if value > best_val:
                best_val = value
                best_x = x
                best_w = w_worst

    # 5) naechster Kontext: Ziehung aus der worst-case-Verteilung an x_t (Simulator-Setting)
    w_normalized = np.clip(best_w, 0.0, None)
    w_normalized /= w_normalized.sum()
    next_context = context_candidates[rng.choice(n_c, p=w_normalized)]

    # Im Kontext-Szenario (config.CONTEXT_SCENARIO_ENABLED) ersetzt
    # core_bo.run_task() diesen Kontext durch c_t ~ P* aus der Umgebung - so wie
    # im Paper vorgesehen. Ohne Kontext-Szenario bleibt die Simulator-Naeherung.
    next_x_full = np.concatenate([best_x, next_context])
    next_x = torch.as_tensor(next_x_full, dtype=torch.double)
    acq_value = torch.as_tensor(best_val, dtype=torch.double)

    return next_x, acq_value, {"posterior_mean": posterior_mean_fn(model)}
