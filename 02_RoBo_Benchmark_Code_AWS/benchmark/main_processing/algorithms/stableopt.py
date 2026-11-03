"""STABLE OPT (Bogunovic et al. 2018) fuer die Pipeline.

Gleiches GP-Modell wie GP_BO (RBF-ARD, LogNormal-Priors), damit sich die
Verfahren nur in der Akquisition unterscheiden. Szenario prior_error: die
Mean-Konstante wird wie bei GP_BO eingefroren.
"""

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction, UpperConfidenceBound
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.utils.transforms import t_batch_mode_transform
from gpytorch.mlls import ExactMarginalLogLikelihood

from preprocessing import config
from main_processing.scenarios.prior_error import apply_prior_error
from main_processing.algorithms.gp_bo import (
    get_covar_module, get_gaussian_likelihood_with_lognormal_prior,
    get_hyperparameters, posterior_mean_fn, posterior_std_fn,
)


def sample_delta_set(dim, epsilon, n_samples, norm, rng):
    # Feste Stoerungsmenge als Diskretisierung von Delta_eps:
    # delta = 0, Achsenpunkte +-eps*e_i, halb Rand (Ecken bzw. Kugeloberflaeche), halb innen
    n_boundary = n_samples // 2
    n_inner = n_samples - n_boundary
    axis = epsilon * np.vstack([np.eye(dim), -np.eye(dim)])

    if norm == "linf":                    # eps-Wuerfel, wie Metrik M3
        boundary = epsilon * rng.choice([-1.0, 1.0], size=(n_boundary, dim))
        inner = rng.uniform(-epsilon, epsilon, size=(n_inner, dim))
    elif norm == "l2":                    # eps-Kugel
        directions = rng.normal(size=(n_samples, dim))
        directions /= np.clip(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12, None)
        radii = rng.uniform(0.0, 1.0, size=(n_inner, 1)) ** (1.0 / dim)
        boundary = epsilon * directions[:n_boundary]
        inner = epsilon * radii * directions[n_boundary:]
    else:
        raise ValueError(f"Unbekannte Norm fuer Delta_eps: {norm}")

    return np.vstack([np.zeros((1, dim)), axis, boundary, inner])


class RobustLCB(AcquisitionFunction):
    # max_j lcb(x + delta_j), negiert, damit optimize_acqf (maximiert) argmin_x findet.
    # x + delta wird auf den Suchraum geschnitten.

    def __init__(self, model, beta, deltas, lower_bound, upper_bound):
        super().__init__(model=model)
        self.beta_sqrt = beta ** 0.5
        self.register_buffer("deltas", torch.as_tensor(deltas, dtype=torch.double))
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        # X: b x 1 x d -> b x m x d, alle Stoerungen je Kandidat in einem Aufruf
        points = (X + self.deltas).clamp(self.lower_bound, self.upper_bound)
        posterior = self.model.posterior(points)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1)
        lcb = mean - self.beta_sqrt * std
        return -lcb.max(dim=-1).values


def run_stable_opt(train_X, train_Y, dim, acquisition_func,
                    lower_bound=config.BOUNDS[0], upper_bound=config.BOUNDS[1],
                    prior_offset=None, rng=None,
                    *, kernel, epsilon, norm, beta, n_delta_candidates,
                    num_restarts, raw_samples):

    if acquisition_func != "LCB":
        raise ValueError(
            f"STABLE OPT verwendet intern feste (ucb,lcb)-Konfidenzbaender; "
            f"unbekannte/inkompatible Akquisitionsfunktion: {acquisition_func}")

    if epsilon is None:                   # gleiche Stoerung wie Metrik M3
        epsilon = config.METRIC_EPSILON

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
    deltas = sample_delta_set(dim, epsilon, n_delta_candidates, norm, rng)

    # 2) x~_t = argmin_x max_j lcb_{t-1}(x + delta_j)
    # Startpunkte: raw_samples Zufallspunkte + Trainingspunkte, die num_restarts besten
    # werden in optimize_acqf lokal verfeinert (gleiche Einstellungen wie GP_BO)
    robust_lcb = RobustLCB(model, beta, deltas, lower_bound, upper_bound)
    raw = torch.as_tensor(rng.uniform(lower_bound, upper_bound, size=(raw_samples, dim)),
                          dtype=torch.double)
    pool = torch.cat([raw, train_X.to(torch.double)]).unsqueeze(1)       # N x 1 x d
    with torch.no_grad():
        pool_values = torch.cat([robust_lcb(chunk) for chunk in pool.split(256)])
    initial = pool[pool_values.topk(min(num_restarts, len(pool))).indices]

    x_tilde, neg_robust_lcb = optimize_acqf(
        robust_lcb, bounds=bounds, q=1, num_restarts=len(initial),
        batch_initial_conditions=initial,
    )
    x_tilde = x_tilde.squeeze(0).detach()

    # 3) delta_t = argmax_{delta in Delta_eps(x~_t)} ucb_{t-1}(x~_t + delta)
    ucb = UpperConfidenceBound(model, beta=beta, maximize=True)
    perturbed = (x_tilde + torch.as_tensor(deltas, dtype=torch.double)).clamp(lower_bound, upper_bound)
    with torch.no_grad():
        ucb_values = ucb(perturbed.unsqueeze(1))
    query_point = perturbed[ucb_values.argmax()]
    worst_ucb = float(ucb_values.max())

    if norm == "linf":
        # eps-Wuerfel um x~_t geschnitten mit X ist ein Quader -> direkt optimierbar
        box = torch.stack([(x_tilde - epsilon).clamp_min(lower_bound),
                           (x_tilde + epsilon).clamp_max(upper_bound)])
        box_point, box_ucb = optimize_acqf(
            ucb, bounds=box, q=1, num_restarts=num_restarts, raw_samples=raw_samples,
        )
        if float(box_ucb) > worst_ucb:
            query_point, worst_ucb = box_point.squeeze(0), float(box_ucb)

    next_x = query_point.detach()
    acq_value = (-neg_robust_lcb).detach()      # robuster lcb-Wert von x~_t

    hyperparameters = get_hyperparameters(model, prior_offset, kernel)
    hyperparameters.update({
        "acq_beta": beta,
        "epsilon": float(epsilon),
        "norm": norm,
        "x_tilde": x_tilde.tolist(),      # robuster Kandidat (Empfehlung laut Paper)
        "robust_ucb": worst_ucb,
        "n_delta_candidates": len(deltas),
        "acq_num_restarts": num_restarts,
        "acq_raw_samples": raw_samples,
    })
    return next_x, acq_value, {
        "posterior_mean": posterior_mean_fn(model),
        "posterior_std": posterior_std_fn(model),
        "hyperparameters": hyperparameters,
        # Empfehlungsregel ROBUST im Runner: x~_T = argmin_t max_delta ucb_{t-1}(x~_t + delta)
        "robust_recommendation": {"x": x_tilde.numpy().astype(np.float64), "score": worst_ucb},
    }
