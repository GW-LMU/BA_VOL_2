import math

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from main_processing.scenarios.prior_error import apply_prior_error

try:
    from botorch.models.utils.gpytorch_modules import (
        get_covar_module_with_dim_scaled_prior,
        get_gaussian_likelihood_with_lognormal_prior,
    )
except ImportError:
    # botorch < 0.12: identischer Nachbau (Hvarfner et al. 2024, wie in botorch 0.12)
    from gpytorch.constraints import GreaterThan
    from gpytorch.kernels import RBFKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.priors import LogNormalPrior

    def _lognormal_mode(loc, scale):
        return math.exp(loc - scale ** 2)

    def get_covar_module_with_dim_scaled_prior(ard_num_dims):
        loc, scale = math.sqrt(2) + 0.5 * math.log(ard_num_dims), math.sqrt(3)
        return RBFKernel(
            ard_num_dims=ard_num_dims,
            lengthscale_prior=LogNormalPrior(loc=loc, scale=scale),
            lengthscale_constraint=GreaterThan(
                2.5e-2, transform=None, initial_value=_lognormal_mode(loc, scale)),
        )

    def get_gaussian_likelihood_with_lognormal_prior():
        return GaussianLikelihood(
            noise_prior=LogNormalPrior(loc=-4.0, scale=1.0),
            noise_constraint=GreaterThan(
                1e-4, transform=None, initial_value=_lognormal_mode(-4.0, 1.0)),
        )


def posterior_mean_fn(model):
    def mean(X_np):
        X = torch.as_tensor(np.atleast_2d(X_np), dtype=torch.double)
        with torch.no_grad():
            return model.posterior(X).mean.squeeze(-1).cpu().numpy()
    return mean


def posterior_std_fn(model):
    # Standardabweichung der latenten Funktion (ohne Beobachtungsrauschen)
    def std(X_np):
        X = torch.as_tensor(np.atleast_2d(X_np), dtype=torch.double)
        with torch.no_grad():
            return model.posterior(X).variance.clamp_min(0).sqrt().squeeze(-1).cpu().numpy()
    return std


def get_hyperparameters(model, prior_offset):
    # Alles, was zusammen mit den Trainingsdaten den Posterior eindeutig festlegt.
    # Laengenskalen beziehen sich auf den normierten Eingaberaum [0, 1]^d (Normalize),
    # Rauschen und Mean-Konstante auf standardisierte y-Werte (Standardize).
    return {
        "kernel": "RBF_ARD",
        "lengthscale": model.covar_module.lengthscale.detach().reshape(-1).tolist(),
        "noise": float(model.likelihood.noise.detach().reshape(-1)[0]),
        "mean_constant": float(model.mean_module.constant.detach()),
        "mean_fixed": prior_offset is not None,
        "y_standardize_mean": float(model.outcome_transform.means.detach().reshape(-1)[0]),
        "y_standardize_std": float(model.outcome_transform.stdvs.detach().reshape(-1)[0]),
    }


def run_gaussian_process(train_X, train_Y, dim, acquisition_func, lower_bound, upper_bound, prior_offset=None):
    gp_bounds = torch.tensor([[lower_bound] * dim, [upper_bound] * dim], dtype=torch.double)
    # Kernel + Likelihood EXPLIZIT gesetzt statt SingleTaskGP-Defaults, da sich
    # diese zwischen BoTorch-Versionen geaendert haben (< 0.12: Matern-5/2 mit
    # Gamma-Priors, >= 0.12: RBF mit dimensionsskaliertem LogNormal-Prior nach
    # Hvarfner et al. 2024). So laeuft auf jeder Instanz dasselbe Modell:
    #   - RBF-Kernel mit ARD (eine Laengenskala je Dimension), LogNormal-Prior
    #     auf die Laengenskalen, skaliert mit sqrt(dim); kein ScaleKernel
    #     (Outputscale entfaellt wegen Standardize)
    #   - Gauss-Likelihood mit LogNormal-Prior auf die Rauschvarianz
    #     (Rauschen wird mitgelernt)
    model = SingleTaskGP(
        train_X, train_Y,
        covar_module=get_covar_module_with_dim_scaled_prior(ard_num_dims=dim),
        likelihood=get_gaussian_likelihood_with_lognormal_prior(),
        input_transform=Normalize(d=dim, bounds=gp_bounds),
        outcome_transform=Standardize(m=1),)


    if prior_offset is not None:          # nur im Szenario prior_error
        apply_prior_error(model, prior_offset)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll) 

    beta = 2.0
    if acquisition_func == "EI":
        acqf = LogExpectedImprovement(model, best_f=train_Y.max())
    elif acquisition_func == "UCB":
        acqf = UpperConfidenceBound(model, beta=beta, maximize=True)
    elif acquisition_func == "LCB":
        acqf = UpperConfidenceBound(model, beta=beta, maximize=False)
    else:
        raise ValueError(f"Unbekannte Akquisitionsfunktion fuer GausianProzess: {acquisition_func}")


    bounds = torch.tensor([[lower_bound] * dim, [upper_bound] * dim], dtype=torch.double)
    num_restarts, raw_samples = 10, 512
    next_x, acq_value = optimize_acqf(
        acqf, bounds=bounds, q=1, num_restarts=num_restarts, raw_samples=raw_samples,
    )

    hyperparameters = get_hyperparameters(model, prior_offset)
    hyperparameters.update({
        "acq_beta": beta if acquisition_func in ("UCB", "LCB") else None,
        "acq_num_restarts": num_restarts,
        "acq_raw_samples": raw_samples,
    })
    return next_x.squeeze(0), acq_value, {
        "posterior_mean": posterior_mean_fn(model),
        "posterior_std": posterior_std_fn(model),
        "hyperparameters": hyperparameters,
    }