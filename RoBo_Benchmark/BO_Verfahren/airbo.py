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
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pfade  # noqa: F401 - fuegt Pre_Processing/ (config) zu sys.path hinzu

import numpy as np
import torch

import config


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
              lower_bound=config.MINUS_AREA, upper_bound=config.PLUS_AREA,
              noise_std=None, n_cloud_samples=50, sub_samp_size=20,
              mmd_length_scale=1.0, mmd_alpha=1.0, noise_var=1e-6,
              beta=2.0, n_x_candidates=200, use_nystrom=True):
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
    train_Y : torch.double, shape [n, 1] (Maximierungs-Konvention, siehe core_bo.py)
    dim     : int, Dimension des Suchraums
    noise_std : Standardabweichung der angenommenen Eingabeunsicherheit
                (Standard: 1% der Suchraumbreite, analog epsilon in stable_opt.py)

    Rueckgabe: next_x (torch.double, shape [dim]), acq_value (torch.double, Skalar),
               extras (dict, "posterior_mean": Funktion fuer core_bo.compute_all_metrics)
    """
    if acquisition_func != "UCB":
        raise ValueError(
            f"AIRBO verwendet intern eine feste UCB-Akquisition auf dem MMD-GP-Posterior; "
            f"unbekannte/inkompatible Akquisitionsfunktion: {acquisition_func}"
        )

    if noise_std is None:
        noise_std = 0.01 * (upper_bound - lower_bound)

    # aus dem globalen numpy-RNG abgeleitet, der in core_bo.run_task() pro Task
    # geseedet wird (config.BO_SEED) -> reproduzierbar
    rng = np.random.default_rng(np.random.randint(2**31 - 1))

    train_X_np = train_X.detach().cpu().numpy()
    train_Y_np = train_Y.squeeze(-1).detach().cpu().numpy()
    n = train_X_np.shape[0]

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

    # 3)+4) Kandidatenraster bewerten: GP-Posterior (Gl. 10/11) + UCB (Gl. 15)
    x_candidates = rng.uniform(lower_bound, upper_bound, size=(n_x_candidates, dim))

    best_ucb = -np.inf
    best_x = None
    for x_cand in x_candidates:
        cand_cloud = _sample_cloud(x_cand, noise_std, n_cloud_samples, lower_bound, upper_bound, rng)
        k_vec = np.array([
            _mmd_kernel_value(cand_cloud, tc, mmd_length_scale, mmd_alpha,
                              use_nystrom, sub_samp_size, rng)
            for tc in train_clouds
        ])
        k_self = _mmd_kernel_value(cand_cloud, cand_cloud, mmd_length_scale, mmd_alpha,
                                    use_nystrom, sub_samp_size, rng)

        mean = float(k_vec @ K_inv @ train_Y_np)
        var = max(k_self - float(k_vec @ K_inv @ k_vec), 0.0)
        ucb = mean + beta * np.sqrt(var)

        if ucb > best_ucb:
            best_ucb = ucb
            best_x = x_cand

    next_x = torch.as_tensor(best_x, dtype=torch.double)
    acq_value = torch.as_tensor(best_ucb, dtype=torch.double)

    # Posterior-Mittelwert fuer core_bo.compute_all_metrics: an den
    # Trainingspunkten direkt aus der bereits berechneten Kernelmatrix
    # (ohne Rauschterm), fuer neue Punkte ueber eine eigene Stichprobenwolke.
    train_means = (K - noise_var * np.eye(n)) @ K_inv @ train_Y_np

    def posterior_mean(X_np):
        X_np = np.atleast_2d(X_np)
        out = np.empty(X_np.shape[0])
        for r, x in enumerate(X_np):
            hit = np.flatnonzero(np.all(np.isclose(train_X_np, x), axis=1))
            if hit.size:
                out[r] = train_means[hit[0]]
                continue
            cloud = _sample_cloud(x, noise_std, n_cloud_samples, lower_bound, upper_bound, rng)
            k = np.array([
                _mmd_kernel_value(cloud, tc, mmd_length_scale, mmd_alpha,
                                  use_nystrom, sub_samp_size, rng)
                for tc in train_clouds
            ])
            out[r] = float(k @ K_inv @ train_Y_np)
        return out

    return next_x, acq_value, {"posterior_mean": posterior_mean}
