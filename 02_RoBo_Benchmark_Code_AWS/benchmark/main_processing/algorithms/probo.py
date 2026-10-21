import numpy as np
import torch

from preprocessing import config
from main_processing.algorithms.uncertainty_aware_bo import BayesOptimizer, matern, rbf


# Kernel-Name aus config.ALGORITHM_CONFIGS -> Kernel-Funktion
_KERNELS = {"Matern32": matern, "RBF": rbf}


def run_imprecise_gausian_prozess(train_X, train_Y, dim, acquisition_func,
                                   lower_bound=config.BOUNDS[0], upper_bound=config.BOUNDS[1],
                                   kernel=config.ALGORITHM_CONFIGS["PROBO"]["kernel"], c=50.0, length_scale=1.0, outputscale=1.0,
                                   n_candidates=500, prior_offset=None, rng=None):
    # rng: np.random.Generator fuer die Kandidaten (reproduzierbar, vom Runner pro Wiederholung)
    # prior_offset: float (Szenario prior_error) oder None
    rng = np.random.default_rng() if rng is None else rng
    kernel = _KERNELS[kernel] if isinstance(kernel, str) else kernel

    y_raw = train_Y.squeeze(-1).detach().cpu().numpy()

    # y standardisieren wie Standardize(m=1) bei GP_BO (unverzerrte Std., Schutz
    # gegen Std. ~ 0). Erst dadurch beziehen sich die Prior-Kandidaten Ms in [0, 5]
    # und c auf eine einheitliche Skala - unabhaengig vom Wertebereich der BBOB-Funktion.
    y_mean = float(np.mean(y_raw))
    y_std = float(np.std(y_raw, ddof=1)) if len(y_raw) > 1 else 1.0
    if not np.isfinite(y_std) or y_std < 1e-8:
        y_std = 1.0
    y_obs = (y_raw - y_mean) / y_std

    # Prior-Fehler (Szenario prior_error): Zentrum der Prior-Mittelwert-Menge um
    # prior_offset verschieben (standardisierte Einheiten, wie bei GP_BO die
    # eingefrorene Mean-Konstante; +1 = optimistisch). Ein GP mit Prior-Mittelwert
    # delta + M*h auf y ist identisch zu einem GP mit Prior-Mittelwert M*h auf
    # y - delta; Vorhersagen werden danach wieder um delta verschoben.
    offset = 0.0 if prior_offset is None else float(prior_offset)
    y_obs = y_obs - offset

    if dim == 1:
        #  Fall dim == 1: unveraendert wie bisher 
        x_obs = train_X.squeeze(-1).detach().cpu().numpy()
        x_candidates = np.linspace(lower_bound, upper_bound, n_candidates)
    else:
        # --- Fall dim > 1: neue, vektorisierte Erweiterung ---
        x_obs = train_X.detach().cpu().numpy()  # shape (n, dim)
        x_candidates = rng.uniform(lower_bound, upper_bound, size=(n_candidates, dim))

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

    # Rueckgabe auf der Skala y = -f (wie GP_BO): Offset zurueck, dann destandardisieren
    def posterior_mean(X_np):
        X_np = np.atleast_2d(np.asarray(X_np, dtype=float))
        X_k = X_np.reshape(-1) if dim == 1 else X_np
        k = bo.kernel(X_k, bo.x_obs)
        return (k @ bo.C_inv @ resid + y_hat + offset) * y_std + y_mean

    # Standardabweichung des Standardposteriors (M=0, h=1), wie in bo.UCB verwendet
    def posterior_std(X_np):
        X_np = np.atleast_2d(np.asarray(X_np, dtype=float))
        X_k = X_np.reshape(-1) if dim == 1 else X_np
        return np.array([np.sqrt(max(bo.predict_posterior(x)[1], 0.0)) for x in X_k]) * y_std

    # Feste Modellparameter (kein Fitten) + daraus abgeleitete Groessen
    hyperparameters = {
        "kernel": kernel.__name__,
        "length_scale": float(length_scale),
        "outputscale": float(outputscale),
        "c": float(c),
        "tau": 1.0,
        "rho": 1.0 if acquisition_func == "GLCB" else None,
        "jitter": 1e-8,
        "n_candidates": int(n_candidates),
        "y_hat": float(y_hat),
        "worst_M": float(bo.worst_M),
        "worst_h": float(bo.worst_h),
        "most_likely_M": float(bo.most_likely_M),
        "most_likely_h": float(bo.most_likely_h),
        # Skalierung/Prior-Fehler: y_modell = (y - y_standardize_mean) / y_standardize_std - prior_offset
        "y_standardize_mean": y_mean,
        "y_standardize_std": y_std,
        "prior_offset": offset,
    }

    return next_x, acq_value, {
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "hyperparameters": hyperparameters,
    }
