"""Robuste Relevance-Pursuit-GP-Variante (Ament et al. 2024) fuer die Pipeline.

Angepasste Kopie von ROBO/Relevance_Pursuit.py. Gegenueber dem Original:
  - Bounds kommen aus config.BOUNDS bzw. vom Runner
  - gleicher Kernel wie GP_BO (RBF-ARD mit dimensionsskaliertem Prior), damit
    sich die Verfahren nur in der Ausreisser-Likelihood unterscheiden
  - Szenario prior_error: Mean-Konstante wird wie bei GP_BO eingefroren
  - Rueckgabe (next_x, acq_value, model_info) wie bei den anderen Verfahren
"""

import torch
from botorch.models.robust_relevance_pursuit_model import RobustRelevancePursuitSingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from preprocessing import config
from main_processing.scenarios.prior_error import apply_prior_error
from main_processing.algorithms.gp_bo import (
    get_covar_module_with_dim_scaled_prior, posterior_mean_fn, posterior_std_fn,
)


def run_relevance_pursuit(train_X, train_Y, dim, acquisition_func,
                          lower_bound=config.BOUNDS[0], upper_bound=config.BOUNDS[1],
                          prior_offset=None):
    """
    Fuehrt die robuste Relevance-Pursuit-GP-Variante fuer eine Iteration aus:
    1) RobustRelevancePursuitSingleTaskGP fitten (erkennt Ausreisser automatisch
       ueber eine datenpunktspezifische Rauschvarianz, siehe Ament et al. 2024)
    2) passende Akquisitionsfunktion anhand von acquisition_func waehlen ("EI", "UCB", "LCB")
    3) naechsten Punkt per optimize_acqf vorschlagen (Zufall ueber den torch-Seed des Runners)

    train_X : torch.double, shape [n, dim]
    train_Y : torch.double, shape [n, 1] (Maximierungs-Konvention, y = -f)
    prior_offset : float (Szenario prior_error) oder None

    Rueckgabe: (next_x, acq_value, model_info) mit posterior_mean/-std und hyperparameters
    """
    bounds = torch.tensor(
        [[lower_bound] * dim, [upper_bound] * dim], dtype=torch.double
    )

    # 1) Modell (Inputs auf [0,1]^d normalisiert, Outputs standardisiert). Die
    #    Likelihood bleibt die Ausreisser-Likelihood des Modells (SparseOutlierNoise)
    model = RobustRelevancePursuitSingleTaskGP(
        train_X, train_Y,
        covar_module=get_covar_module_with_dim_scaled_prior(ard_num_dims=dim),
        input_transform=Normalize(d=dim, bounds=bounds),
        outcome_transform=Standardize(m=1),
    )

    if prior_offset is not None:          # nur im Szenario prior_error
        apply_prior_error(model, prior_offset)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # 2) Akquisitionsfunktion abfragen und passend waehlen
    beta = 2.0
    if acquisition_func == "EI":
        acqf = LogExpectedImprovement(model, best_f=train_Y.max())
    elif acquisition_func == "UCB":
        acqf = UpperConfidenceBound(model, beta=beta, maximize=True)
    elif acquisition_func == "LCB":
        acqf = UpperConfidenceBound(model, beta=beta, maximize=False)
    else:
        raise ValueError(f"Unbekannte Akquisitionsfunktion fuer Relevance Pursuit: {acquisition_func}")

    # 3) naechsten Punkt vorschlagen
    num_restarts, raw_samples = 10, 512
    next_x, acq_value = optimize_acqf(
        acqf, bounds=bounds, q=1, num_restarts=num_restarts, raw_samples=raw_samples,
    )

    noise_covar = model.likelihood.noise_covar
    hyperparameters = {
        "kernel": "RBF_ARD",
        "lengthscale": model.covar_module.lengthscale.detach().reshape(-1).tolist(),
        "noise": float(noise_covar.base_noise.noise.detach().reshape(-1)[0]),
        # Indizes der als Ausreisser erkannten Trainingspunkte + deren Zusatzvarianz
        "outlier_support": [int(i) for i in noise_covar.support],
        "outlier_rho": noise_covar.rho.detach().reshape(-1).tolist(),
        "mean_constant": float(model.mean_module.constant.detach()),
        "mean_fixed": prior_offset is not None,
        "y_standardize_mean": float(model.outcome_transform.means.detach().reshape(-1)[0]),
        "y_standardize_std": float(model.outcome_transform.stdvs.detach().reshape(-1)[0]),
        "acq_beta": beta if acquisition_func in ("UCB", "LCB") else None,
        "acq_num_restarts": num_restarts,
        "acq_raw_samples": raw_samples,
    }

    return next_x.squeeze(0), acq_value, {
        "posterior_mean": posterior_mean_fn(model),
        "posterior_std": posterior_std_fn(model),
        "hyperparameters": hyperparameters,
    }
