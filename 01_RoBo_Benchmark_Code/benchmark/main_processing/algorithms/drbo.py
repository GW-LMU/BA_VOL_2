"""Distributionally-Robust-BO-Variante (Kirschner et al. 2020) fuer die verteilte Pipeline.

Angepasste Kopie von ROBO/DRBO.py. Gegenueber dem Original:
  - Bounds kommen aus config.BOUNDS bzw. vom Runner
  - gleiches GP-Modell wie GP_BO (RBF-ARD, LogNormal-Priors)
  - Szenario prior_error: Mean-Konstante wird wie bei GP_BO eingefroren
  - Zufall ueber den rng des Runners, Rueckgabe (next_x, acq_value, model_info)
"""

import numpy as np
import scipy.optimize
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from preprocessing import config
from main_processing.scenarios.prior_error import apply_prior_error
from main_processing.algorithms.gp_bo import (
    get_covar_module, get_gaussian_likelihood_with_lognormal_prior,
    get_hyperparameters, posterior_mean_fn, posterior_std_fn,
)


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
             lower_bound=config.BOUNDS[0], upper_bound=config.BOUNDS[1],
             prior_offset=None, rng=None, *, kernel, context_dim, n_context_candidates, n_x_candidates,
             mmd_lengthscale, delta, beta):
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
    train_Y : torch.double, shape [n, 1] (Maximierungs-Konvention, y = -f)
    dim     : int, Dimension des Gesamtraums (Kontrollvariablen + Kontextvariablen)
    prior_offset : float (Szenario prior_error) oder None
    rng     : np.random.Generator fuer Kandidaten und Kontext-Ziehung (vom Runner pro Wiederholung)
    kernel, context_dim, ..., beta : aus config.ALGORITHM_CONFIGS["DRBO"]

    Rueckgabe: (next_x, acq_value, model_info)
    """

    # Hier prüfen ob richtige Acqistionsfunktion gewählt worden ist
    # ("UCB" = distributionally robust UCB-Akquisition, siehe config)
    if acquisition_func != "UCB":
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
        covar_module=get_covar_module(kernel, ard_num_dims=dim),
        likelihood=get_gaussian_likelihood_with_lognormal_prior(),
        input_transform=Normalize(d=dim, bounds=bounds),
        outcome_transform=Standardize(m=1),
    )

    if prior_offset is not None:          # nur im Szenario prior_error
        apply_prior_error(model, prior_offset)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Hier fitte ich das Modell damit ich die  Define (ucbtx)j := ˆft(x,cj) + βtσt(x,cj) bekomme
    # Fitten über RKHS Regression 

    rng = np.random.default_rng() if rng is None else rng

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

    next_x_full = np.concatenate([best_x, next_context])
    next_x = torch.as_tensor(next_x_full, dtype=torch.double)
    acq_value = torch.as_tensor(best_val, dtype=torch.double)

    hyperparameters = get_hyperparameters(model, prior_offset, kernel)
    hyperparameters.update({
        "acq_beta": beta,
        "context_dim": context_dim,
        "epsilon_t": float(epsilon_t),
        "delta": delta,
        "mmd_lengthscale": mmd_lengthscale,
        "n_context_candidates": n_context_candidates,
        "n_x_candidates": n_x_candidates,
    })
    return next_x, acq_value, {
        "posterior_mean": posterior_mean_fn(model),
        "posterior_std": posterior_std_fn(model),
        "hyperparameters": hyperparameters,
    }
