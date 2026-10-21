
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pfade  # noqa: F401 - fuegt Pre_Processing/ (config) zu sys.path hinzu

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_lognormal_prior,
)
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np


def posterior_mean_fn(model):
    """Gibt eine Funktion X_np -> Posterior-Mittelwert (Originalskala, numpy)
    fuer ein BoTorch-Modell zurueck. Wird von core_bo.run_task() benutzt, um den
    empfohlenen Punkt x_rec zu bestimmen (siehe compute_all_metrics)."""
    def mean(X_np):
        X = torch.as_tensor(np.atleast_2d(X_np), dtype=torch.double)
        with torch.no_grad():
            return model.posterior(X).mean.squeeze(-1).cpu().numpy()
    return mean


def run_gaussian_process(train_X, train_Y, dim, acquisition_func, lower_bound, upper_bound):
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
    
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    if acquisition_func == "EI":
        acqf = LogExpectedImprovement(model, best_f=train_Y.max())
    elif acquisition_func == "UCB":
        acqf = UpperConfidenceBound(model, beta=2.0, maximize=True)
    elif acquisition_func == "LCB":
        acqf = UpperConfidenceBound(model, beta=2.0, maximize=False)
    else:
        raise ValueError(f"Unbekannte Akquisitionsfunktion fuer GausianProzess: {acquisition_func}")
   

    bounds = torch.tensor([[lower_bound] * dim, [upper_bound] * dim], dtype=torch.double)
    next_x, acq_value = optimize_acqf(
        acqf, bounds=bounds, q=1, num_restarts=10, raw_samples=512,
    )
    return next_x.squeeze(0), acq_value, {"posterior_mean": posterior_mean_fn(model)}
