import numpy as np
import torch
from scipy.stats import qmc

from preprocessing import config
from main_processing.algorithms.uncertainty_aware_bo import BayesOptimizer, matern, rbf


# Kernel-Name aus config.ALGORITHM_CONFIGS -> Kernel-Funktion
_KERNELS = {"Matern32": matern, "RBF": rbf}


def focus_search(af_batch, lower_bound, upper_bound, dim, rng,
                 n_points=1000, maxit=5, restarts=5):
    """Maximiert eine Akquisitionsfunktion per Focus Search (wie mlrMBO).

    Nachbau von mlrMBO::optimizeFocusSearch (Bischl et al. 2017, arXiv:1703.03373,
    S. 7), dem AF-Optimierer aus Rodemann & Augustin (2024), Abschnitt 6:
    "Focus search [...] with 1000 evaluations per round and 5 maximal restarts".

    Ablauf je Restart (Suchbox startet jeweils wieder beim vollen Suchraum):
      fuer maxit Runden:
        1) n_points Punkte per Latin Hypercube in der aktuellen Box ziehen
        2) Akquisition auswerten, global bestes x merken
        3) Box um das beste x dieser Runde verkleinern: pro Dimension
           [max(lo, x - Breite/4), min(hi, x + Breite/4)], also halbe Breite

    af_batch: Funktion X -> Akquisitionswerte (shape (N,)), wird MAXIMIERT
    rng: np.random.Generator (reproduzierbar, vom Runner pro Wiederholung)
    Rueckgabe: (bestes x als shape (dim,), bester Akquisitionswert)
    """
    best_x, best_val = None, -np.inf
    for _ in range(restarts):
        lo = np.full(dim, float(lower_bound))
        hi = np.full(dim, float(upper_bound))
        for _ in range(maxit):
            # mlrMBO zieht die Punkte als zufaelliges LHS-Design
            X = qmc.scale(qmc.LatinHypercube(d=dim, seed=rng).random(n_points), lo, hi)
            vals = af_batch(X[:, 0] if dim == 1 else X)
            idx = int(np.argmax(vals))
            if vals[idx] > best_val:
                best_x, best_val = X[idx].copy(), float(vals[idx])
            width = hi - lo
            lo = np.maximum(lo, X[idx] - width / 4)
            hi = np.minimum(hi, X[idx] + width / 4)
    return best_x, best_val


def run_imprecise_gausian_prozess(train_X, train_Y, dim, acquisition_func,
                                   lower_bound=config.BOUNDS[0], upper_bound=config.BOUNDS[1],
                                   kernel=config.ALGORITHM_CONFIGS["PROBO"]["kernel"], c=50.0, length_scale=1.0, outputscale=1.0,
                                   focus_points=1000, focus_maxit=5, focus_restarts=5,
                                   prior_offset=None, rng=None):
    # rng: np.random.Generator fuer die Focus Search (reproduzierbar, vom Runner pro Wiederholung)
    # focus_*: Einstellungen der Focus Search, siehe focus_search()
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
        x_obs = train_X.squeeze(-1).detach().cpu().numpy()  # shape (n,)
    else:
        x_obs = train_X.detach().cpu().numpy()  # shape (n, dim)

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

    # 2) Akquisitionsfunktion abfragen und passend waehlen (vektorisierte Varianten,
    #    identische Formeln wie bo.UCB / bo.PROBO)
    if acquisition_func == "LCB":
        af_batch = bo.UCB_batch
    elif acquisition_func == "GLCB":
        af_batch = bo.PROBO_batch
        # Hier mit festen tau=1.0, rho=1.0
    else:
        raise ValueError(
            f"Unbekannte Akquisitionsfunktion fuer Imprecise_GausianProzess_Rodemann: {acquisition_func}"
        )

    # 3) Akquisition per Focus Search maximieren (wie Rodemann & Augustin 2024)
    best_x, best_val = focus_search(af_batch, lower_bound, upper_bound, dim, rng,
                                    n_points=focus_points, maxit=focus_maxit,
                                    restarts=focus_restarts)

    next_x = torch.tensor(best_x, dtype=torch.double)  # shape (dim,), bei dim=1 also (1,)
    acq_value = torch.tensor(best_val, dtype=torch.double)

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

    # Standardabweichung des Standardposteriors (M=0, h=1), wie in bo.UCB verwendet.
    # Vektorisiert: dieselbe Formel wie predict_posterior(x, M=0, h=1), aber fuer
    # alle Punkte auf einmal statt einer Schleife (Abweichung zur Schleife <= 2e-10)
    def posterior_std(X_np):
        X_np = np.atleast_2d(np.asarray(X_np, dtype=float))
        X_k = X_np.reshape(-1) if dim == 1 else X_np
        K, kxx, quad = bo._kernel_terms(X_k)
        var = kxx - quad + (1 - K @ bo.s_k) ** 2 / (bo.c + bo.S_k)
        return np.sqrt(np.maximum(var, 0.0)) * y_std

    # Feste Modellparameter (kein Fitten) + daraus abgeleitete Groessen
    hyperparameters = {
        "kernel": kernel.__name__,
        "length_scale": float(length_scale),
        "outputscale": float(outputscale),
        "c": float(c),
        "tau": 1.0,
        "rho": 1.0 if acquisition_func == "GLCB" else None,
        "jitter": 1e-8,
        "af_optimizer": "focus_search",
        "focus_points": int(focus_points),
        "focus_maxit": int(focus_maxit),
        "focus_restarts": int(focus_restarts),
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
