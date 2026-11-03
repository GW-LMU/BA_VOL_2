import numpy as np
import torch
from scipy.stats import qmc

from preprocessing import config
from main_processing.algorithms.uncertainty_aware_bo import BayesOptimizer, matern, rbf


# Hintergrund zu Hyperparameter-Fit, Prior-Fehler und Wahl von c = 1:
# PROBO_Hyperparameter_und_Prior_Fehler.md (im selben Ordner)

# Kernel-Name aus config.ALGORITHM_CONFIGS["PROBO"] -> Kernel-Funktion
_KERNELS = {"Matern32": matern, "RBF": rbf}

# Rauschvarianz beim Fitten: BayesOptimizer rechnet rauschfrei (nur Jitter 1e-8 in
# robust_inverse), GPyTorch braucht fuer eine stabile Cholesky aber etwas mehr.
_FIT_NOISE = 1e-6


def fit_kernel_hyperparameters(train_X, y_std_scale, dim, kernel_name,
                               lower_bound, upper_bound, prior_mean):
    """Schaetzt length_scale und outputscale des PROBO-Kernels per Empirical Bayes.

    Damit werden die Kernel-Hyperparameter wie bei GP_BO, DRBO, StableOpt und RP_BO
    in jeder Iteration per Marginal Likelihood gefittet (vorher: fest 1.0 / 1.0).

    Modell (passend zu BayesOptimizer in uncertainty_aware_bo.py):
      - isotroper Kernel (EINE Laengenskala, kein ARD), Matern-3/2 oder RBF,
        mit Outputscale (ScaleKernel) - dieselben Formeln wie matern()/rbf()
      - y bereits standardisiert (kein Standardize-Transform)
      - Rauschen fest auf _FIT_NOISE, da BayesOptimizer rauschfrei rechnet
      - konstanter Prior-Mittelwert EINGEFROREN auf prior_mean (wie im Szenario
        prior_error bei den anderen Verfahren, apply_prior_error). Im nominalen
        Szenario ist prior_mean = 0, also genau der Mittelwert des Standard-
        posteriors von PROBO (M = 0). Im Szenario prior_error gilt prior_mean = delta:
        Die Hyperparameter werden unter dem falschen Prior geschaetzt und koennen den
        Mittelwertfehler dadurch teilweise auffangen (wie bei GP_BO). Der Effekt des
        Prior-Fehlers ist also delta inklusive Anpassung des Kernels, nicht delta allein.
      - Laengenskalen-Prior wie GP_BO (dimensionsskalierter LogNormal-Prior nach
        Hvarfner et al. 2024), dafuer werden die Eingaben auf [0, 1] normalisiert

    Rueckgabe: (length_scale, outputscale) auf der Skala von BayesOptimizer, d. h.
    die Laengenskala im ORIGINALEN Suchraum [lower_bound, upper_bound].
    """
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.input import Normalize
    from gpytorch.constraints import GreaterThan
    from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.priors import LogNormalPrior

    from main_processing.scenarios.prior_error import apply_prior_error

    loc, scale = np.sqrt(2) + 0.5 * np.log(dim), np.sqrt(3)
    base_kwargs = dict(
        lengthscale_prior=LogNormalPrior(loc=loc, scale=scale),
        lengthscale_constraint=GreaterThan(2.5e-2, transform=None,
                                           initial_value=float(np.exp(loc - scale ** 2))),
    )
    if kernel_name == "Matern32":
        base = MaternKernel(nu=1.5, **base_kwargs)
    elif kernel_name == "RBF":
        base = RBFKernel(**base_kwargs)
    else:
        raise ValueError(f"Unbekannter PROBO-Kernel: {kernel_name} (erlaubt: {', '.join(_KERNELS)})")

    X = train_X.detach().to(torch.double).reshape(-1, dim)
    Y = torch.as_tensor(y_std_scale, dtype=torch.double).reshape(-1, 1)
    gp_bounds = torch.tensor([[lower_bound] * dim, [upper_bound] * dim], dtype=torch.double)

    model = SingleTaskGP(
        X, Y, train_Yvar=torch.full_like(Y, _FIT_NOISE),
        covar_module=ScaleKernel(base),
        input_transform=Normalize(d=dim, bounds=gp_bounds),
        outcome_transform=None,                       # y ist schon standardisiert
    )
    apply_prior_error(model, prior_mean)              # Mittelwert einfrieren, VOR dem Fit
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

    # Normalize skaliert jede Dimension auf [0, 1] -> Laengenskala zurueck in den Suchraum
    length_scale = float(base.lengthscale.detach()) * (upper_bound - lower_bound)
    outputscale = float(model.covar_module.outputscale.detach())
    return length_scale, outputscale


def focus_search(af_batch, lower_bound, upper_bound, dim, rng,
                 n_points=1000, maxit=5, restarts=5):
    """Minimiert eine Akquisitionsfunktion per Focus Search (wie mlrMBO).

    Nachbau von mlrMBO::optimizeFocusSearch (Bischl et al. 2017, arXiv:1703.03373,
    S. 7), dem AF-Optimierer aus Rodemann & Augustin (2024), Abschnitt 6:
    "Focus search [...] with 1000 evaluations per round and 5 maximal restarts".

    Ablauf je Restart (Suchbox startet jeweils wieder beim vollen Suchraum):
      fuer maxit Runden:
        1) n_points Punkte per Latin Hypercube in der aktuellen Box ziehen
        2) Akquisition auswerten, global bestes x merken
        3) Box um das beste x dieser Runde verkleinern: pro Dimension
           [max(lo, x - Breite/4), min(hi, x + Breite/4)], also halbe Breite

    af_batch: Funktion X -> Akquisitionswerte (shape (N,)), wird MINIMIERT
    rng: np.random.Generator (reproduzierbar, vom Runner pro Wiederholung)
    Rueckgabe: (bestes x als shape (dim,), bester Akquisitionswert)
    """
    best_x, best_val = None, np.inf
    for _ in range(restarts):
        lo = np.full(dim, float(lower_bound))
        hi = np.full(dim, float(upper_bound))
        for _ in range(maxit):
            # mlrMBO zieht die Punkte als zufaelliges LHS-Design
            X = qmc.scale(qmc.LatinHypercube(d=dim, seed=rng).random(n_points), lo, hi)
            vals = af_batch(X[:, 0] if dim == 1 else X)
            idx = int(np.argmin(vals))
            if vals[idx] < best_val:
                best_x, best_val = X[idx].copy(), float(vals[idx])
            width = hi - lo
            lo = np.maximum(lo, X[idx] - width / 4)
            hi = np.minimum(hi, X[idx] + width / 4)
    return best_x, best_val


def run_imprecise_gausian_prozess(train_X, train_Y, dim, acquisition_func,
                                   lower_bound=config.BOUNDS[0], upper_bound=config.BOUNDS[1],
                                   prior_offset=None, rng=None,
                                   *, kernel, c, fit_hyperparameters, length_scale, outputscale,
                                   prior_mean_max, n_prior_means,
                                   tau, rho, focus_points, focus_maxit, focus_restarts):
    # kernel, c, ..., focus_restarts: aus config.ALGORITHM_CONFIGS["PROBO"]
    # fit_hyperparameters: True -> length_scale/outputscale per Empirical Bayes fitten
    #   (fit_kernel_hyperparameters), False -> feste Werte length_scale/outputscale
    # prior_mean_max, n_prior_means: Prior-Mittelwert-Menge M in [-prior_mean_max, prior_mean_max]
    # tau, rho: Gewichte von Varianzterm und Credal-Set-Breite in der Akquisition
    # rng: np.random.Generator fuer die Focus Search (reproduzierbar, vom Runner pro Wiederholung)
    # focus_*: Einstellungen der Focus Search, siehe focus_search()
    # prior_offset: float (Szenario prior_error) oder None
    rng = np.random.default_rng() if rng is None else rng
    kernel_name = kernel if isinstance(kernel, str) else next(
        name for name, fn in _KERNELS.items() if fn is kernel)
    kernel = _KERNELS[kernel_name]

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
    # eingefrorene Mean-Konstante; -1 = optimistisch). Ein GP mit Prior-Mittelwert
    # delta + M*h auf y ist identisch zu einem GP mit Prior-Mittelwert M*h auf
    # y - delta; Vorhersagen werden danach wieder um delta verschoben.
    offset = 0.0 if prior_offset is None else float(prior_offset)

    # Kernel-Hyperparameter fitten (wie die anderen Verfahren), auf den
    # standardisierten y mit eingefrorenem Prior-Mittelwert offset. Das entspricht
    # genau dem Modell, das BayesOptimizer unten bekommt (y - offset, Mittelwert 0);
    # das spaetere Vorzeichen (-y) aendert die Marginal Likelihood nicht.
    if fit_hyperparameters:
        length_scale, outputscale = fit_kernel_hyperparameters(
            train_X, y_obs, dim, kernel_name, lower_bound, upper_bound, prior_mean=offset)

    y_obs = y_obs - offset

    # Die Pipeline minimiert (y = f), BayesOptimizer (uncertainty_aware_bo.py,
    # bewusst unveraendert) rechnet UCB/PROBO aber fuer Maximierung. Deshalb
    # bekommt er -y und die Akquisition wird unten zurueck-negiert:
    #   -UCB(-y)   = mu - tau*var             = LCB auf y
    #   -PROBO(-y) = mu - tau*var - rho*Breite = GLCB auf y (Def. 11)
    # Die Prior-Menge M in [-prior_mean_max, prior_mean_max] ist symmetrisch,
    # das Vorzeichen aendert an ihr also nichts.
    y_obs = -y_obs

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
        prior_mean_max=prior_mean_max,
        n_prior_means=n_prior_means,
    )

    # 2) Akquisitionsfunktion abfragen und passend waehlen (vektorisierte Varianten,
    #    identische Formeln wie bo.UCB / bo.PROBO, negiert -> LCB / GLCB auf y = f)
    if acquisition_func == "LCB":
        af_batch = lambda X: -bo.UCB_batch(X, tau=tau)
    elif acquisition_func == "GLCB":
        af_batch = lambda X: -bo.PROBO_batch(X, tau=tau, rho=rho)
    else:
        raise ValueError(
            f"Unbekannte Akquisitionsfunktion fuer Imprecise_GausianProzess_Rodemann: {acquisition_func}"
        )

    # 3) Akquisition per Focus Search minimieren (wie Rodemann & Augustin 2024)
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

    # Rueckgabe auf der Skala y = f (wie GP_BO): Vorzeichen zurueck, Offset zurueck,
    # dann destandardisieren
    def posterior_mean(X_np):
        X_np = np.atleast_2d(np.asarray(X_np, dtype=float))
        X_k = X_np.reshape(-1) if dim == 1 else X_np
        k = bo.kernel(X_k, bo.x_obs)
        return (-(k @ bo.C_inv @ resid + y_hat) + offset) * y_std + y_mean

    # Standardabweichung des Standardposteriors (M=0, h=1), wie in bo.UCB verwendet.
    # Vektorisiert: dieselbe Formel wie predict_posterior(x, M=0, h=1), aber fuer
    # alle Punkte auf einmal statt einer Schleife (Abweichung zur Schleife <= 2e-10)
    def posterior_std(X_np):
        X_np = np.atleast_2d(np.asarray(X_np, dtype=float))
        X_k = X_np.reshape(-1) if dim == 1 else X_np
        K, kxx, quad = bo._kernel_terms(X_k)
        var = kxx - quad + (1 - K @ bo.s_k) ** 2 / (bo.c + bo.S_k)
        return np.sqrt(np.maximum(var, 0.0)) * y_std

    # Modellparameter (length_scale/outputscale gefittet oder fest, je nach
    # fit_hyperparameters) + daraus abgeleitete Groessen
    hyperparameters = {
        "kernel": kernel.__name__,
        "hyperparameters_fitted": bool(fit_hyperparameters),
        "length_scale": float(length_scale),
        "outputscale": float(outputscale),
        "c": float(c),
        "prior_mean_max": float(prior_mean_max),
        "n_prior_means": int(n_prior_means),
        "tau": float(tau),
        "rho": float(rho) if acquisition_func == "GLCB" else None,
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
        # Skalierung/Prior-Fehler: y_modell = -((y - y_standardize_mean) / y_standardize_std - prior_offset)
        "y_standardize_mean": y_mean,
        "y_standardize_std": y_std,
        "prior_offset": offset,
    }

    return next_x, acq_value, {
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "hyperparameters": hyperparameters,
    }
