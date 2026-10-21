"""AIRBO-Variante (Yang, Lyu, Lyu, Chen 2023, NeurIPS - "Efficient Robust Bayesian
Optimization for Arbitrary Uncertain inputs") fuer die verteilte Pipeline.

Eigenstaendige, kompakte Neuimplementierung nach den Formeln im Paper (nicht
der schwergewichtige GPyTorch-Produktionscode aus huawei-noah/HEBO/AIRBO, der
eine eigene ExactGP-Unterklasse mit Latent-Mapping-MLP, Batch-Nystroem-Kernel
ueber mehrere Module umfasst und nicht in das einheitliche run_*()-Schema
dieses Projekts passt). Diese Version implementiert den fachlichen Kern:

  - MMD-Kernel ueber Eingabeverteilungen statt exakten Punkten (Sec. 3.1,
    Gl. 8/9): k_hat(P,Q) = exp(-alpha * MMD^2(P,Q))
  - Nystroem-Approximation zur Beschleunigung der MMD-Schaetzung (Sec. 3.2,
    Gl. 12), reduziert die Kosten von O(m^2) auf O(m*h) mit h << m
  - GP-Posterior ueber diesen Kernel (Gl. 10/11) + UCB-Akquisition (Gl. 15)

Wichtige Vereinfachung/Annahme gegenueber dem Paper: run_task() in core_bo.py
liefert nur EXAKTE Beobachtungen (x_i, y_i), keine echte Eingabeunsicherheits-
verteilung P_xi. Diese Implementierung nimmt daher an, dass um jeden
beobachteten (bzw. vorgeschlagenen) Punkt eine feste, isotrope Gauss-
Unsicherheit N(x, noise_std^2 * I) liegt (analog zur "execution noise" aus
der Motivation des Papers, Sec. 1) und approximiert P_x durch n_cloud_samples
Stichproben daraus. Der zurueckgegebene next_x ist der Zentrumspunkt der
robust besten Kandidatenwolke (nicht eine einzelne Stichprobe daraus), damit
er sich wie bei den anderen Varianten direkt und exakt auswerten laesst.

Anpassung an die Pipeline: y wird wie bei PROBO standardisiert (der GP hat
Prior-Mittelwert 0). Im Szenario prior_error wird der Prior-Mittelwert um
prior_offset verschoben (gleicher Trick wie in probo.py).
"""

import numpy as np
import torch

from preprocessing import config


def _rbf_gram(A, B, length_scale):
    """Paarweise RBF-Kernelmatrix (Basiskernel fuer die MMD-Schaetzung, Gl. 8)."""
    sqdist = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T
    sqdist = np.clip(sqdist, 0.0, None)
    return np.exp(-0.5 * sqdist / length_scale**2)


def _sample_cloud(center, noise_std, n_samples, lower_bound, upper_bound, rng):
    """Zieht n_samples Stichproben aus N(center, noise_std^2 * I) als Naeherung
    fuer die (unbekannte) Eingabeunsicherheitsverteilung P_center, begrenzt auf
    den Suchraum."""
    dim = center.shape[0]
    samples = center[None, :] + rng.normal(scale=noise_std, size=(n_samples, dim))
    return np.clip(samples, lower_bound, upper_bound)


def _empirical_mmd2(U, V, length_scale):
    """Unverzerrter empirischer MMD^2-Schaetzer zwischen zwei Stichprobenwolken
    U, V (je m x dim) ueber den RBF-Basiskernel (Gl. 8)."""
    m, h = U.shape[0], V.shape[0]
    k_uu = _rbf_gram(U, U, length_scale)
    k_vv = _rbf_gram(V, V, length_scale)
    k_uv = _rbf_gram(U, V, length_scale)
    term_uu = (k_uu.sum() - np.trace(k_uu)) / (m * (m - 1))
    term_vv = (k_vv.sum() - np.trace(k_vv)) / (h * (h - 1))
    term_uv = k_uv.sum() / (m * h)
    return term_uu + term_vv - 2.0 * term_uv


def _nystrom_mmd2(U, V, length_scale, sub_samp_size, rng):
    """Nystroem-Schaetzer fuer MMD^2 (Gl. 12): approximiert die Kernelmatrizen
    ueber h << m zufaellige Teilstichproben statt der vollen m x m-Matrix."""
    h_u = min(sub_samp_size, U.shape[0])
    h_v = min(sub_samp_size, V.shape[0])
    U_sub = U[rng.choice(U.shape[0], size=h_u, replace=False)]
    V_sub = V[rng.choice(V.shape[0], size=h_v, replace=False)]

    k_m_u = _rbf_gram(U_sub, U_sub, length_scale)
    k_mn_u = _rbf_gram(U_sub, U, length_scale)
    alpha_u = (np.linalg.pinv(k_m_u) @ k_mn_u @ np.ones(U.shape[0])) / U.shape[0]

    k_m_v = _rbf_gram(V_sub, V_sub, length_scale)
    k_mn_v = _rbf_gram(V_sub, V, length_scale)
    alpha_v = (np.linalg.pinv(k_m_v) @ k_mn_v @ np.ones(V.shape[0])) / V.shape[0]

    part1 = alpha_u @ k_m_u @ alpha_u
    part2 = alpha_v @ k_m_v @ alpha_v
    part3 = alpha_u @ _rbf_gram(U_sub, V_sub, length_scale) @ alpha_v
    return float(part1 + part2 - 2.0 * part3)


def _mmd_kernel_value(U, V, length_scale, alpha, use_nystrom, sub_samp_size, rng):
    """k_hat(P,Q) = exp(-alpha * MMD^2(P,Q)), Gl. 9."""
    if use_nystrom:
        mmd2 = _nystrom_mmd2(U, V, length_scale, sub_samp_size, rng)
    else:
        mmd2 = _empirical_mmd2(U, V, length_scale)
    return float(np.exp(-alpha * max(mmd2, 0.0)))


def run_airbo(train_X, train_Y, dim, acquisition_func,
              lower_bound=config.BOUNDS[0], upper_bound=config.BOUNDS[1],
              prior_offset=None, rng=None, *, noise_std, n_cloud_samples, sub_samp_size,
              mmd_length_scale, mmd_alpha, noise_var, beta, n_x_candidates, use_nystrom):
    """
    Fuehrt eine Iteration von AIRBO aus (Yang et al. 2023, Algorithmus gemaess
    Sec. 3 + Gl. 15):

        1) fuer jeden bisherigen Beobachtungspunkt eine Stichprobenwolke aus der
           angenommenen Eingabeunsicherheit ziehen (siehe Modulkommentar)
        2) MMD-Kernelmatrix ueber diese Wolken bilden (Gl. 8/9, optional per
           Nystroem-Approximation beschleunigt, Gl. 12)
        3) GP-Posterior ueber diesem Kernel bilden (Gl. 10/11)
        4) fuer ein Kandidatenraster im Suchraum den UCB-Wert berechnen
           (Gl. 15) und den besten Kandidaten zurueckgeben

    train_X : torch.double, shape [n, dim]
    train_Y : torch.double, shape [n, 1] (Maximierungs-Konvention, y = -f)
    dim     : int, Dimension des Suchraums
    prior_offset : float (Szenario prior_error) oder None
    rng     : np.random.Generator fuer Wolken, Nystroem und Kandidaten (vom Runner pro Wiederholung)
    noise_std : Standardabweichung der angenommenen Eingabeunsicherheit
                (None: 1% der Suchraumbreite, analog epsilon in stableopt.py)
    noise_std, ..., use_nystrom : aus config.ALGORITHM_CONFIGS["AIRBO"]

    Rueckgabe: (next_x, acq_value, model_info); acq_value ist der UCB-Wert auf der
    standardisierten Modellskala
    """
    if acquisition_func != "UCB":
        raise ValueError(
            f"AIRBO verwendet intern eine feste UCB-Akquisition auf dem MMD-GP-Posterior; "
            f"unbekannte/inkompatible Akquisitionsfunktion: {acquisition_func}"
        )

    if noise_std is None:
        noise_std = 0.01 * (upper_bound - lower_bound)

    rng = np.random.default_rng() if rng is None else rng

    train_X_np = train_X.detach().cpu().numpy()
    y_raw = train_Y.squeeze(-1).detach().cpu().numpy()
    n = train_X_np.shape[0]

    # y standardisieren wie bei PROBO/Standardize(m=1) (der GP hat Prior-Mittelwert 0)
    y_mean = float(np.mean(y_raw))
    y_std = float(np.std(y_raw, ddof=1)) if n > 1 else 1.0
    if not np.isfinite(y_std) or y_std < 1e-8:
        y_std = 1.0

    # Prior-Fehler (Szenario prior_error): GP mit Prior-Mittelwert delta auf y ist
    # identisch zu GP mit Mittelwert 0 auf y - delta (wie in probo.py)
    offset = 0.0 if prior_offset is None else float(prior_offset)
    train_Y_np = (y_raw - y_mean) / y_std - offset

    # 1) Stichprobenwolken um die bisherigen Beobachtungen
    train_clouds = [
        _sample_cloud(x, noise_std, n_cloud_samples, lower_bound, upper_bound, rng)
        for x in train_X_np
    ]

    # 2) MMD-Kernelmatrix zwischen allen Trainingswolken (Gl. 8/9/12)
    K = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            val = _mmd_kernel_value(
                train_clouds[i], train_clouds[j], mmd_length_scale, mmd_alpha,
                use_nystrom, sub_samp_size, rng,
            )
            K[i, j] = K[j, i] = val
    K += noise_var * np.eye(n)
    K_inv = np.linalg.pinv(K)
    weights = K_inv @ train_Y_np

    def cloud_posterior(x, gen):
        """GP-Posterior (Gl. 10/11) an der Wolke um x, standardisierte Modellskala."""
        cloud = _sample_cloud(x, noise_std, n_cloud_samples, lower_bound, upper_bound, gen)
        k_vec = np.array([
            _mmd_kernel_value(cloud, tc, mmd_length_scale, mmd_alpha,
                              use_nystrom, sub_samp_size, gen)
            for tc in train_clouds
        ])
        k_self = _mmd_kernel_value(cloud, cloud, mmd_length_scale, mmd_alpha,
                                   use_nystrom, sub_samp_size, gen)
        mean = float(k_vec @ weights)
        var = max(k_self - float(k_vec @ K_inv @ k_vec), 0.0)
        return mean, var

    # 3)+4) Kandidatenraster bewerten: GP-Posterior (Gl. 10/11) + UCB (Gl. 15)
    x_candidates = rng.uniform(lower_bound, upper_bound, size=(n_x_candidates, dim))

    best_ucb = -np.inf
    best_x = None
    for x_cand in x_candidates:
        mean, var = cloud_posterior(x_cand, rng)
        ucb = mean + offset + beta * np.sqrt(var)

        if ucb > best_ucb:
            best_ucb = ucb
            best_x = x_cand

    next_x = torch.as_tensor(best_x, dtype=torch.double)
    acq_value = torch.as_tensor(best_ucb, dtype=torch.double)

    # Posterior fuer die Rekonstruktion im Runner (Skala y = -f). Eigener Seed,
    # damit posterior_mean und posterior_std dieselben Wolken verwenden und das
    # Protokollieren den Zufallsstrom des Verfahrens nicht veraendert.
    posterior_seed = int(rng.integers(2**32))

    def _posterior(X_np):
        gen = np.random.default_rng(posterior_seed)
        results = [cloud_posterior(x, gen) for x in np.atleast_2d(np.asarray(X_np, dtype=float))]
        means = np.array([m for m, _ in results])
        stds = np.sqrt(np.array([v for _, v in results]))
        return (means + offset) * y_std + y_mean, stds * y_std

    hyperparameters = {
        "kernel": "MMD_RBF",
        "noise_std": float(noise_std),
        "n_cloud_samples": n_cloud_samples,
        "sub_samp_size": sub_samp_size,
        "use_nystrom": use_nystrom,
        "mmd_length_scale": mmd_length_scale,
        "mmd_alpha": mmd_alpha,
        "noise_var": noise_var,
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