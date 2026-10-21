"""STABLE OPT (Bogunovic et al. 2018) fuer die Pipeline.

Gleiches GP-Modell wie GP_BO (RBF-ARD, LogNormal-Priors), damit sich die
Verfahren nur in der Akquisition unterscheiden. Szenario prior_error: die
Mean-Konstante wird wie bei GP_BO eingefroren.
"""

import numpy as np
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


def sample_ball(center, epsilon, n_samples, lower_bound, upper_bound, rng):
    """
    Zieht n_samples Punkte aus der euklidischen Kugel mit Radius epsilon um `center`
    (Delta_epsilon(x) aus Bogunovic et al. 2018), beschraenkt auf die Suchraumgrenzen.
    """
    dim = center.shape[0]
    directions = rng.normal(size=(n_samples, dim))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.clip(norms, 1e-12, None)
    radii = epsilon * rng.uniform(0.0, 1.0, size=(n_samples, 1)) ** (1.0 / dim)
    points = center[None, :] + directions * radii
    return np.clip(points, lower_bound, upper_bound)


def run_stable_opt(train_X, train_Y, dim, acquisition_func,
                    lower_bound=config.BOUNDS[0], upper_bound=config.BOUNDS[1],
                    prior_offset=None, rng=None,
                    *, kernel, epsilon, beta, n_x_candidates, n_delta_candidates):
    """
    Fuehrt eine Iteration des STABLE OPT-Algorithmus aus (Bogunovic et al. 2018,
    "Adversarially Robust Optimization with Gaussian Processes", Algorithm 1):

        x~_t = argmax_x  min_{delta in Delta_eps(x)}  ucb_{t-1}(x + delta)      (13)
        delta_t = argmin_{delta in Delta_eps(x~_t)}  lcb_{t-1}(x~_t + delta)     (Zeile 3)
        query: x~_t + delta_t

    Delta_eps(x) ist die euklidische epsilon-Kugel um x (d(x,x')=||x-x'||_2), geschnitten
    mit dem Suchraum. Da die verschachtelte Min-Max-Optimierung auf kontinuierlichem X
    im Allgemeinen nicht exakt loesbar ist (siehe Paper, Abschnitt 3), wird sie hier per
    Kandidaten-Sampling approximiert, analog zum num_restarts/raw_samples-Vorgehen der
    uebrigen Surrogatmodelle in diesem Projekt.

    train_X : torch.double, shape [n, dim]
    train_Y : torch.double, shape [n, 1] (Maximierungs-Konvention, y = -f)
    dim     : int, Dimension des Suchraums
    prior_offset : float (Szenario prior_error) oder None
    rng     : np.random.Generator fuer das Kandidaten-Sampling (vom Runner pro Wiederholung)
    epsilon : Stabilitaetsradius (None: 1% der Suchraumbreite)
    kernel, epsilon, beta, n_x_candidates, n_delta_candidates : aus config.ALGORITHM_CONFIGS["STABLEOPT"]

    Rueckgabe: (next_x, acq_value, model_info); acq_value ist der ucb-Wert von x~_t
    """
    if acquisition_func != "UCB":
        raise ValueError(
            f"STABLE OPT verwendet intern feste (ucb,lcb)-Konfidenzbaender; "
            f"unbekannte/inkompatible Akquisitionsfunktion: {acquisition_func}")

    if epsilon is None:
        epsilon = 0.01 * (upper_bound - lower_bound)

    bounds = torch.tensor(
        [[lower_bound] * dim, [upper_bound] * dim], dtype=torch.double
    )

    # 1) Modell fitten (Inputs auf [0,1]^d normalisiert, Outputs standardisiert)
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

    rng = np.random.default_rng() if rng is None else rng

    def ucb_lcb(points_np):
        points = torch.as_tensor(points_np, dtype=torch.double)
        with torch.no_grad():
            posterior = model.posterior(points)
            mean = posterior.mean.squeeze(-1)
            std = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1)
        beta_sqrt = beta ** 0.5
        
        ucb = (mean + beta_sqrt * std).numpy()

        lcb = (mean - beta_sqrt * std).numpy()
        return ucb, lcb

    # 2) x~_t = argmax_x min_{delta in Delta_eps(x)} ucb_{t-1}(x+delta)  (Kandidaten-Sampling)
    x_candidates = rng.uniform(lower_bound, upper_bound, size=(n_x_candidates, dim))

    best_stable_ucb = -np.inf
    x_tilde = None
    for x in x_candidates:
        perturbed = sample_ball(x, epsilon, n_delta_candidates, lower_bound, upper_bound, rng)
        ucb_vals, _ = ucb_lcb(perturbed)
        stable_ucb = ucb_vals.min()
        if stable_ucb  > best_stable_ucb:
            best_stable_ucb = stable_ucb
            x_tilde = x

    # 3) delta_t = argmin_{delta in Delta_eps(x~_t)} lcb_{t-1}(x~_t + delta)
    perturbed_final = sample_ball(x_tilde, epsilon, n_delta_candidates, lower_bound, upper_bound, rng)
    _, lcb_vals = ucb_lcb(perturbed_final)
    delta_idx = np.argmin(lcb_vals)
    query_point = perturbed_final[delta_idx]

    next_x = torch.as_tensor(query_point, dtype=torch.double)
    acq_value = torch.as_tensor(best_stable_ucb, dtype=torch.double)

    hyperparameters = get_hyperparameters(model, prior_offset, kernel)
    hyperparameters.update({
        "acq_beta": beta,
        "epsilon": float(epsilon),
        "x_tilde": x_tilde.tolist(),      # robuster Kandidat (Empfehlung laut Paper)
        "n_x_candidates": n_x_candidates,
        "n_delta_candidates": n_delta_candidates,
    })
    return next_x, acq_value, {
        "posterior_mean": posterior_mean_fn(model),
        "posterior_std": posterior_std_fn(model),
        "hyperparameters": hyperparameters,
    }


"""
StableOpt: Notizen

Idee des Verfahrens (Bogunovic et al. 2018)

Gesucht ist ein Punkt x, der auch dann noch gut ist, wenn er um bis zu ε verschoben wird (Störung δ mit ‖δ‖ ≤ ε).
Schritt 1 (robuster Kandidat): Für jedes x wird der schlechteste UCB-Wert in seiner ε-Umgebung bestimmt. Gewählt wird das x, bei dem dieser Wert am höchsten ist:
x̃ = argmax_x min_δ UCB(x+δ)
Schritt 2 (Auswertungspunkt): Innerhalb der ε-Umgebung von x̃ wird die Störung δ mit dem niedrigsten LCB-Wert gewählt:
δ = argmin_δ LCB(x̃+δ)
Ausgewertet wird x̃ + δ, nicht x̃ selbst. So wird gezielt geprüft, ob x̃ wirklich robust ist.

Umsetzung im Code (Näherung)

Nicht jedes x lässt sich exakt prüfen. Deshalb werden Kandidaten gezogen:
256 zufällige x-Kandidaten im Suchraum
pro Kandidat 32 zufällige Punkte in der ε-Kugel, um den schlechtesten UCB-Wert zu schätzen
Der schlechteste Fall wird dadurch nur geschätzt und fällt tendenziell zu optimistisch aus.

Problem bei der Bewertung (Simple Regret)

Der Benchmark bewertet die ausgewerteten Punkte, also x̃ + δ.
Diese Punkte sind absichtlich ungünstig gewählt: Dort ist LCB minimal, das Modell also am pessimistischsten.
Deshalb schneidet StableOpt beim Simple Regret systematisch schlechter ab. Das heißt nicht, dass jeder Punkt der schlechteste ist, sondern dass die Kennzahl nicht misst, was StableOpt optimiert.
Empfehlung aus dem Paper:
Als Ergebnis wird x̃T = argmax_t minδ LCB(x̃_t+δ) empfohlen, also der Kandidat, dessen schlechtester Fall am höchsten ist.
Bewertet wird der robuste Regret: max_x minδ f(x+δ) − minδ f(x̃_T+δ)

Offene Punkte (To-do)

Seed setzen, damit die Läufe reproduzierbar sind. Bisher ist der Zufallsgenerator ohne Seed.
Das Kandidaten-Sampling durch optimize_acqf ersetzen, mit einer festen δ-Menge und denselben Einstellungen wie GP-EI (num_restarts/raw_samples).
ε an das Rauschen NOISE_SIGMA anpassen. Bisher ist ε = 0,1, das Rauschen hat aber σ = 0,2.
x̃ zusätzlich zurückgeben und eine robuste Kennzahl speichern.

Korrektur zu deinem Text:

Ausgewertet wird der Punkt mit dem schlechtesten LCB-Wert in der Umgebung von x̃, nicht der mit dem schlechtesten UCB-Wert. Der schlechteste UCB-Wert wird nur für die Auswahl von x̃ gebraucht.
Der Simple Regret ist deshalb nicht „immer der schlechteste Wert“, sondern systematisch zu pessimistisch für StableOpt.
 
"""