

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pfade  # noqa: F401 - fuegt Pre_Processing/ (config) zu sys.path hinzu

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

import config# ✅
from gaussian_process import posterior_mean_fn


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
                    lower_bound=config.MINUS_AREA, upper_bound=config.PLUS_AREA,
                    epsilon=None, beta=2.0, n_x_candidates=256, n_delta_candidates=32):
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
    train_Y : torch.double, shape [n, 1] (Maximierungs-Konvention, siehe core_bo.py)
    dim     : int, Dimension des Suchraums
    epsilon : Stabilitaetsradius (Standard: 1% der Suchraumbreite)

    Rueckgabe: next_x (torch.double, shape [dim]), acq_value (torch.double, Skalar; ucb-Wert von x~_t),
               extras (dict: posterior_mean, x_tilde, x_tilde_min_lcb - siehe core_bo.run_task)
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
        input_transform=Normalize(d=dim, bounds=bounds),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # aus dem globalen numpy-RNG abgeleitet, der in core_bo.run_task() pro Task
    # geseedet wird (config.BO_SEED) -> reproduzierbar
    rng = np.random.default_rng(np.random.randint(2**31 - 1))

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

    # Fuer die Empfehlungsregel aus dem Paper (x~(T) = x~_t* mit
    # t* = argmax_t min_delta lcb_{t-1}(x~_t + delta)) braucht core_bo.run_task()
    # pro Runde x~_t und den zugehoerigen pessimistischen Wert min_delta lcb.
    extras = {
        "posterior_mean": posterior_mean_fn(model),
        "x_tilde": np.asarray(x_tilde, dtype=float),
        "x_tilde_min_lcb": float(lcb_vals.min()),
    }
    return next_x, acq_value, extras


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