
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pfade  # noqa: F401 - fuegt Pre_Processing/ (config) zu sys.path hinzu

import numpy as np
import torch

import config
from uncertainty_aware_bo import BayesOptimizer, matern


def run_imprecise_gausian_prozess(train_X, train_Y, dim, acquisition_func,
                                   lower_bound=config.MINUS_AREA, upper_bound=config.PLUS_AREA,
                                   kernel=matern, c=50.0, length_scale=1.0, outputscale=1.0,
                                   n_candidates=500):
    
    y_obs = train_Y.squeeze(-1).detach().cpu().numpy()

    if dim == 1:
        #  Fall dim == 1: unveraendert wie bisher 
        x_obs = train_X.squeeze(-1).detach().cpu().numpy()
        x_candidates = np.linspace(lower_bound, upper_bound, n_candidates)
    else:
        # --- Fall dim > 1: neue, vektorisierte Erweiterung ---
        x_obs = train_X.detach().cpu().numpy()  # shape (n, dim)
        x_candidates = np.random.uniform(lower_bound, upper_bound, size=(n_candidates, dim))

    # 1) Modell mit den bisherigen Beobachtungen aufbauen
    bo = BayesOptimizer(
        x_obs=x_obs,
        y_obs=y_obs,
        kernel=kernel,
        c=c,
        length_scale=length_scale,
        outputscale=outputscale,
        dim=dim,
    )

    # 2) Akquisitionsfunktion abfragen und passend waehlen
    if acquisition_func == "LCB":
        af = bo.UCB
    elif acquisition_func == "GLCB":
        af = bo.PROBO
        # Hier mit festen tau=1.0, rho=1.0
    else:
        raise ValueError(
            f"Unbekannte Akquisitionsfunktion fuer Imprecise_GausianProzess_Rodemann: {acquisition_func}"
        )

    # 3) Kandidaten auswerten, bestes x auswaehlen
    acquisition_values = np.array([af(x) for x in x_candidates])
    best_idx = np.argmax(acquisition_values)

    if dim == 1:
        next_x = torch.tensor([x_candidates[best_idx]], dtype=torch.double)
    else:
        next_x = torch.tensor(x_candidates[best_idx], dtype=torch.double)
    acq_value = torch.tensor(acquisition_values[best_idx], dtype=torch.double)

    # Posterior-Mittelwert fuer core_bo.compute_all_metrics - jeweils DERSELBE
    # Mittelwert, den die gewaehlte Akquisition verwendet:
    #   "GLCB" -> PROBO.mean_term = k(x)^T C^-1 y
    #   "LCB"  -> UCB -> predict_posterior(x, M=0, h=1)
    if acquisition_func == "LCB":
        y_hat = float(bo.s_k @ bo.y_obs) / (bo.c + bo.S_k)
        resid = bo.y_obs - y_hat
    else:
        y_hat = 0.0
        resid = bo.y_obs

    def posterior_mean(X_np):
        X_np = np.atleast_2d(np.asarray(X_np, dtype=float))
        X_k = X_np.reshape(-1) if dim == 1 else X_np
        k = bo.kernel(X_k, bo.x_obs)
        return k @ bo.C_inv @ resid + y_hat

    return next_x, acq_value, {"posterior_mean": posterior_mean}
