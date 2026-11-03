"""AIRBO-Variante (Yang, Lyu, Lyu, Chen 2023, NeurIPS - "Efficient Robust Bayesian
Optimization for Arbitrary Uncertain inputs") fuer die verteilte Pipeline.

Eigenstaendige, kompakte Neuimplementierung nach den Formeln im Paper (nicht
der schwergewichtige GPyTorch-Produktionscode aus huawei-noah/HEBO/AIRBO, der
eine eigene ExactGP-Unterklasse mit Latent-Mapping-MLP, Batch-Nystroem-Kernel
ueber mehrere Module umfasst und nicht in das einheitliche run_*()-Schema
dieses Projekts passt). Diese Version implementiert den fachlichen Kern:

  - MMD-Kernel ueber Eingabeverteilungen statt exakten Punkten (Sec. 3.1,
    Gl. 8/9): k_hat(P,Q) = s * exp(-alpha * MMD^2(P,Q)), Basiskernel RBF-ARD
  - Nystroem-Approximation der Kernel-Mean-Embeddings (Sec. 3.2, Gl. 12),
    reduziert die Kosten von O(m^2) auf O(m*h) mit h << m
  - Hyperparameter (Laengenskalen des Basiskernels, alpha, Outputscale s,
    Beobachtungsrauschen) per (MAP-)Marginal-Likelihood gelernt (Sec. 3.1)
  - GP-Posterior ueber diesen Kernel (Gl. 10/11) + LCB-Akquisition (Gl. 15 in
    Minimierungs-Form; das Paper maximiert die UCB)

Eingabeunsicherheit: um jeden beobachteten (bzw. vorgeschlagenen) Punkt x liegt
P_x = N(x, noise_std^2 * I), approximiert durch n_cloud_samples Stichproben. Wie
im Paper wird das Modell mit (x, y) trainiert, wobei y im Szenario
stochastic_input an x + xi mit xi ~ P_x - x ausgewertet wurde (siehe
scenarios/stochastic_input.py). noise_std wird vom Runner gesetzt (bekannte
Eingabeverteilung wie im Paper, siehe experiment_runner.py).

Damit der MMD-Kernel deterministisch und positiv semidefinit ist, werden alle
Wolken innerhalb einer Iteration mit denselben Basisstichproben gebildet
(cloud(x) = x + sigma * E, Common Random Numbers) und jede Wolke hat genau EIN
festes Nystroem-Embedding (feste Landmarks). MMD ist dann ein Abstand im RKHS,
exp(-alpha * MMD^2) ein gueltiger Kernel und k_hat(P,P) = s exakt.

Anpassung an die Pipeline: Eingaben auf [0,1]^d normiert (wie Normalize bei
GP_BO), y standardisiert (der GP hat Prior-Mittelwert 0). Im Szenario
prior_error wird der Prior-Mittelwert um prior_offset verschoben (gleicher Trick
wie in probo.py).
"""

import math

import numpy as np
import torch

from preprocessing import config

_JITTER = 1e-6                      # Regularisierung der Landmark-Gram-Matrix (Nystroem)
_LENGTHSCALE_MIN = 2.5e-2           # wie GP_BO (normierter Eingaberaum)
_NOISE_MIN = 1e-4                   # wie GP_BO (standardisierte y)


def _rbf(A, B, lengthscale):
    """RBF-ARD-Basiskernel (Gl. 8) zwischen den letzten beiden Achsen von A und B."""
    A = A / lengthscale
    B = B / lengthscale
    sqdist = (A**2).sum(-1)[..., :, None] + (B**2).sum(-1)[..., None, :] - 2 * A @ B.transpose(-1, -2)
    return torch.exp(-0.5 * sqdist.clamp_min(0.0))


def _embeddings(clouds, n_landmarks, lengthscale, use_nystrom):
    """Kernel-Mean-Embeddings der Wolken (b x m x d) als Gewichte a (b x h) auf
    Landmarks Z (b x h x d): mu_P ~ sum_k a_k phi(z_k).
    Nystroem (Gl. 12): a = K_ZZ^+ K_ZX 1 / m mit Z = erste h Stichproben jeder Wolke
    (die Stichproben sind iid, das entspricht einer festen Zufallsauswahl).
    Ohne Nystroem: Z = alle Stichproben, a = 1/m (empirisches Embedding)."""
    m = clouds.shape[1]
    if not use_nystrom or n_landmarks >= m:
        return clouds, torch.full(clouds.shape[:2], 1.0 / m, dtype=clouds.dtype)
    Z = clouds[:, :n_landmarks]
    K_zz = _rbf(Z, Z, lengthscale) + _JITTER * torch.eye(n_landmarks, dtype=clouds.dtype)
    K_zx_mean = _rbf(Z, clouds, lengthscale).mean(-1, keepdim=True)
    return Z, torch.linalg.solve(K_zz, K_zx_mean).squeeze(-1)


def _embedding_inner(Z1, a1, Z2, a2, lengthscale):
    """<mu_i, mu_j>_RKHS fuer alle Paare (i aus 1, j aus 2) -> (b1 x b2)."""
    b1, h1, d = Z1.shape
    b2, h2, _ = Z2.shape
    G = _rbf(Z1.reshape(b1 * h1, d), Z2.reshape(b2 * h2, d), lengthscale).reshape(b1, h1, b2, h2)
    return torch.einsum("ih,ihjk,jk->ij", a1, G, a2)


def _self_inner(Z, a, lengthscale):
    """||mu_i||^2 fuer jede Wolke einzeln -> (b,)."""
    return torch.einsum("bh,bhk,bk->b", a, _rbf(Z, Z, lengthscale), a)


def _lognormal_log_prob(value, loc, scale):
    log_v = torch.log(value)
    return (-log_v - 0.5 * ((log_v - loc) / scale) ** 2).sum()


def run_airbo(train_X, train_Y, dim, acquisition_func,
              lower_bound=config.BOUNDS[0], upper_bound=config.BOUNDS[1],
              prior_offset=None, rng=None, *, noise_std, n_cloud_samples, sub_samp_size,
              beta, n_x_candidates, use_nystrom, fit_max_iter):
    """
    Fuehrt eine Iteration von AIRBO aus (Yang et al. 2023, Algorithmus gemaess
    Sec. 3 + Gl. 15):

        1) fuer jeden bisherigen Beobachtungspunkt eine Stichprobenwolke aus der
           Eingabeunsicherheit P_x bilden (gemeinsame Basisstichproben)
        2) Kernel-Mean-Embeddings (Nystroem, Gl. 12) und MMD-Kernel (Gl. 8/9)
        3) Hyperparameter per Marginal Likelihood lernen, GP-Posterior (Gl. 10/11)
        4) fuer ein Kandidatenraster im Suchraum den LCB-Wert der Kandidatenwolke
           berechnen (Gl. 15, Minimierungs-Form) und den kleinsten zurueckgeben

    train_X : torch.double, shape [n, dim]
    train_Y : torch.double, shape [n, 1] (Minimierungs-Konvention, y = f)
    dim     : int, Dimension des Suchraums
    prior_offset : float (Szenario prior_error) oder None
    rng     : np.random.Generator fuer Basisstichproben und Kandidaten (vom Runner pro Wiederholung)
    noise_std : Standardabweichung der Eingabeunsicherheit auf der Originalskala
                (vom Runner gesetzt, siehe experiment_runner.py)
    noise_std, ..., fit_max_iter : aus config.ALGORITHM_CONFIGS["AIRBO"]

    Rueckgabe: (next_x, acq_value, model_info); acq_value ist der LCB-Wert auf der
    standardisierten Modellskala
    """
    if acquisition_func != "LCB":
        raise ValueError(
            f"AIRBO verwendet intern eine feste LCB-Akquisition auf dem MMD-GP-Posterior; "
            f"unbekannte/inkompatible Akquisitionsfunktion: {acquisition_func}"
        )
    if noise_std is None:
        raise ValueError("AIRBO: noise_std muss gesetzt sein (wird im experiment_runner bestimmt).")

    rng = np.random.default_rng() if rng is None else rng
    dtype = torch.double
    width = upper_bound - lower_bound

    # Normierter Eingaberaum [0,1]^d, Eingabeunsicherheit entsprechend skaliert
    U_train = (train_X.detach().to(dtype) - lower_bound) / width
    sigma_u = float(noise_std) / width
    n = U_train.shape[0]

    y_raw = train_Y.squeeze(-1).detach().cpu().numpy()
    # y standardisieren wie bei PROBO/Standardize(m=1) (der GP hat Prior-Mittelwert 0)
    y_mean = float(np.mean(y_raw))
    y_std = float(np.std(y_raw, ddof=1)) if n > 1 else 1.0
    if not np.isfinite(y_std) or y_std < 1e-8:
        y_std = 1.0

    # Prior-Fehler (Szenario prior_error): GP mit Prior-Mittelwert delta auf y ist
    # identisch zu GP mit Mittelwert 0 auf y - delta (wie in probo.py)
    offset = 0.0 if prior_offset is None else float(prior_offset)
    y = torch.as_tensor((y_raw - y_mean) / y_std - offset, dtype=dtype)

    # 1) Gemeinsame Basisstichproben E ~ N(0, I): cloud(x) = x + sigma * E
    E = torch.as_tensor(rng.normal(size=(n_cloud_samples, dim)), dtype=dtype)

    def clouds_of(U):
        return U[:, None, :] + sigma_u * E[None, :, :]

    train_clouds = clouds_of(U_train)
    h = min(sub_samp_size, n_cloud_samples)

    # Hyperparameter: param = untere Schranke + exp(raw). Priors wie GP_BO fuer
    # Laengenskalen (dimensionsskaliert) und Rauschen, schwach informativ fuer alpha, s
    ls_loc, ls_scale = math.sqrt(2) + 0.5 * math.log(dim), math.sqrt(3)
    init = {
        "lengthscale": torch.full((dim,), math.exp(ls_loc - ls_scale**2), dtype=dtype),
        "alpha": torch.tensor(1.0, dtype=dtype),
        "outputscale": torch.tensor(1.0, dtype=dtype),
        "noise": torch.tensor(math.exp(-4.0 - 1.0), dtype=dtype),
    }
    lower = {"lengthscale": _LENGTHSCALE_MIN, "alpha": 1e-4, "outputscale": 1e-4, "noise": _NOISE_MIN}
    raw = {k: torch.log(v - lower[k]).clone().requires_grad_(True) for k, v in init.items()}

    def params():
        return {k: lower[k] + torch.exp(raw[k]) for k in raw}

    def train_kernel(p):
        Z, a = _embeddings(train_clouds, h, p["lengthscale"], use_nystrom)
        S = _embedding_inner(Z, a, Z, a, p["lengthscale"])
        diag = torch.diagonal(S)
        mmd2 = (diag[:, None] + diag[None, :] - 2.0 * S).clamp_min(0.0)
        mmd2 = mmd2 * (1.0 - torch.eye(n, dtype=dtype))      # MMD(P,P) = 0 exakt
        return p["outputscale"] * torch.exp(-p["alpha"] * mmd2), Z, a, diag

    def neg_log_posterior():
        p = params()
        K, _, _, _ = train_kernel(p)
        K = K + (p["noise"] + 1e-8) * torch.eye(n, dtype=dtype)
        L = torch.linalg.cholesky(K)
        v = torch.cholesky_solve(y[:, None], L).squeeze(-1)
        mll = -0.5 * y @ v - torch.log(torch.diagonal(L)).sum() - 0.5 * n * math.log(2 * math.pi)
        log_prior = (_lognormal_log_prob(p["lengthscale"], ls_loc, ls_scale)
                     + _lognormal_log_prob(p["noise"], -4.0, 1.0)
                     + _lognormal_log_prob(p["alpha"], 0.0, 1.0)
                     + _lognormal_log_prob(p["outputscale"], 0.0, 1.0))
        return -(mll + log_prior) / n

    # 3) Hyperparameter per MAP-Marginal-Likelihood (L-BFGS); bei numerischem
    # Fehlschlag werden die Startwerte verwendet
    optimizer = torch.optim.LBFGS(list(raw.values()), max_iter=fit_max_iter,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = neg_log_posterior()
        loss.backward()
        return loss

    start_state = {k: v.detach().clone() for k, v in raw.items()}
    try:
        optimizer.step(closure)
        if not all(torch.isfinite(v).all() for v in raw.values()):
            raise RuntimeError("nicht-endliche Hyperparameter")
        fit_ok = True
    except (RuntimeError, torch.linalg.LinAlgError):
        with torch.no_grad():
            for k in raw:
                raw[k].copy_(start_state[k])
        fit_ok = False

    with torch.no_grad():
        p = params()
        K, Z_train, a_train, self_train = train_kernel(p)
        K = K + (p["noise"] + 1e-8) * torch.eye(n, dtype=dtype)
        L = torch.linalg.cholesky(K)
        weights = torch.cholesky_solve(y[:, None], L).squeeze(-1)

    def cloud_posterior(U, chunk=256):
        """GP-Posterior (Gl. 10/11) an den Wolken um U (b x d, normiert), Modellskala."""
        means, variances = [], []
        with torch.no_grad():
            for start in range(0, U.shape[0], chunk):
                Z, a = _embeddings(clouds_of(U[start:start + chunk]), h, p["lengthscale"], use_nystrom)
                cross = _embedding_inner(Z, a, Z_train, a_train, p["lengthscale"])
                self_c = _self_inner(Z, a, p["lengthscale"])
                mmd2 = (self_c[:, None] + self_train[None, :] - 2.0 * cross).clamp_min(0.0)
                k_vec = p["outputscale"] * torch.exp(-p["alpha"] * mmd2)
                means.append(k_vec @ weights)
                v = torch.linalg.solve_triangular(L, k_vec.T, upper=False)
                variances.append((p["outputscale"] - (v**2).sum(0)).clamp_min(0.0))
        return torch.cat(means).numpy(), torch.cat(variances).numpy()

    # 4) Kandidatenraster bewerten: GP-Posterior (Gl. 10/11) + LCB (Gl. 15, Minimierung)
    x_candidates = rng.uniform(lower_bound, upper_bound, size=(n_x_candidates, dim))
    mean, var = cloud_posterior(torch.as_tensor((x_candidates - lower_bound) / width, dtype=dtype))
    lcb = mean + offset - beta * np.sqrt(var)
    best = int(np.argmin(lcb))

    next_x = torch.as_tensor(x_candidates[best], dtype=dtype)
    acq_value = torch.as_tensor(lcb[best], dtype=dtype)

    # Posterior fuer die Rekonstruktion im Runner (Skala y = f); deterministisch,
    # da Basisstichproben und Hyperparameter fest sind
    def _posterior(X_np):
        U = (torch.as_tensor(np.atleast_2d(np.asarray(X_np, dtype=float)), dtype=dtype) - lower_bound) / width
        m, v = cloud_posterior(U)
        return (m + offset) * y_std + y_mean, np.sqrt(v) * y_std

    hyperparameters = {
        "kernel": "MMD_RBF_ARD",
        # Laengenskalen beziehen sich auf den normierten Eingaberaum [0, 1]^d,
        # Outputscale und Rauschen auf standardisierte y-Werte
        "lengthscale": p["lengthscale"].reshape(-1).tolist(),
        "mmd_alpha": float(p["alpha"]),
        "outputscale": float(p["outputscale"]),
        "noise": float(p["noise"]),
        "fit_ok": fit_ok,
        "noise_std": float(noise_std),
        "n_cloud_samples": n_cloud_samples,
        "sub_samp_size": sub_samp_size,
        "use_nystrom": use_nystrom,
        "acq_beta": beta,
        "n_x_candidates": n_x_candidates,
        # Skalierung/Prior-Fehler: y_modell = (y - y_standardize_mean) / y_standardize_std - prior_offset
        "y_standardize_mean": y_mean,
        "y_standardize_std": y_std,
        "prior_offset": offset,
    }

    return next_x, acq_value, {
        "posterior_mean": lambda X_np: _posterior(X_np)[0],
        "posterior_std": lambda X_np: _posterior(X_np)[1],
        "hyperparameters": hyperparameters,
    }
