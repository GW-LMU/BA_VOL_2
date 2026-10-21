"""Kern des Imprecise-Gaussian-Process-Modells (Rodemann) - schlanke Kopie.

Angepasste/gekuerzte Version von ROBO/uncertainty_aware_bo.py: enthaelt nur
die Teile, die imprecise_gp.py fuer eine BO-Iteration tatsaechlich braucht
(Kernel, BayesOptimizer mit UCB/PROBO-Akquisition). Die Diagnose-/Plot-Methode
`plot_motiv` sowie ungenutzte Akquisitionsvarianten (RAHBO, CredalAF, ...) und
die matplotlib-Abhaengigkeit wurden bewusst entfernt, damit ROBO_NEU ohne
Abhaengigkeit zum alten Ordner auf jede LRZ-Instanz kopiert werden kann.

Mehrdimensionale Erweiterung (dim>1): Die urspruenglichen Kernel `matern`/`rbf`
arbeiten mit `np.subtract.outer` auf skalaren x-Werten und funktionieren daher
nur fuer dim=1. Fuer dim>1 gibt es die vektorisierten Gegenstuecke
`matern_nd`/`rbf_nd` (uebernommen aus homo_bo_compare_af.ipynb, Zelle 19), die
auf Punkten in R^d arbeiten. `BayesOptimizer` waehlt anhand von `dim` beim
Erzeugen automatisch die passende Variante - siehe dortiger Kommentar.
"""

import numpy as np


def matern(x, y, length_scale=1.0, outputscale=1.0):
    """Matern-3/2-Kernel fuer dim=1 (x, y: 1D-Arrays skalarer Punkte)."""
    dists = np.abs(np.subtract.outer(x, y))
    sqrt3_dists = np.sqrt(3) * dists / length_scale
    return outputscale * (1.0 + sqrt3_dists) * np.exp(-sqrt3_dists)


def rbf(x, y, length_scale=1.0, outputscale=1.0):
    """RBF-Kernel fuer dim=1 (x, y: 1D-Arrays skalarer Punkte)."""
    sqdist = np.subtract.outer(x, y) ** 2
    return outputscale * np.exp(-0.5 * sqdist / length_scale ** 2)


def matern_nd(X, Y, length_scale=1.0, outputscale=1.0):
    """Matern-3/2-Kernel fuer dim>1 (X, Y: Arrays von Punkten in R^d).

    X: shape (n, d) oder (d,) fuer einen einzelnen Punkt
    Y: shape (m, d) oder (d,)
    Rueckgabe: Kernelmatrix shape (n, m), analog zu matern() fuer dim=1.
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    dists = np.sqrt(np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1))
    sqrt3_dists = np.sqrt(3) * dists / length_scale
    return outputscale * (1.0 + sqrt3_dists) * np.exp(-sqrt3_dists)


def rbf_nd(X, Y, length_scale=1.0, outputscale=1.0):
    """RBF-Kernel fuer dim>1 (X, Y: Arrays von Punkten in R^d), analog zu rbf()."""
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    sqdist = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
    return outputscale * np.exp(-0.5 * sqdist / length_scale ** 2)


# Ordnet jedem 1D-Kernel sein mehrdimensionales Gegenstueck zu, damit
# BayesOptimizer anhand von `dim` automatisch die passende Variante waehlt.
_ND_KERNELS = {matern: matern_nd, rbf: rbf_nd}


class BayesOptimizer:
    """Imprecise-GP-Modell (Rodemann): robuste Akquisition ueber eine Menge
    unsicherer Priori-Mittelwerte M*h statt eines einzelnen Priors.

    Unterstuetzt dim=1 (urspruengliche, skalare Kernel-Distanzen) und dim>1
    (vektorisierte Kernel ueber R^d, siehe matern_nd/rbf_nd oben). Alle
    Methoden nach __init__ (worst-case-/most-likely-Prior, MLL, Posterior,
    UCB/PROBO) sind fuer beide Faelle identisch, da sie ausschliesslich ueber
    self.kernel (dim-abhaengig gewaehlt) auf x_obs zugreifen.
    """

    def __init__(self, x_obs, y_obs, y_obs_vars=None, kernel=matern, c=50.0,
                 length_scale=1.0, outputscale=1.0, dim=1):
        """
        x_obs: array der beobachteten x-Werte (dim=1: shape (n,); dim>1: shape (n, dim))
        y_obs: array der beobachteten y-Werte
        y_obs_vars: array der beobachteten y-Varianz (aleatorische Unsicherheit),
                    None im Fall homoskedastischen Rauschens
        kernel: 1D-Kernel-Funktion (matern/rbf); fuer dim>1 wird automatisch
                das zugehoerige Gegenstueck aus _ND_KERNELS verwendet
        c: Imprecision-Parameter
        length_scale, outputscale: Kernel-Hyperparameter
        dim: Dimension des Suchraums
        """
        self.dim = dim
        self.y_obs = np.array(y_obs)
        self.y_obs_vars = np.array(y_obs_vars) if y_obs_vars is not None else None
        self.c = c

        if self.dim == 1:
            # --- Fall dim == 1: unveraendert wie bisher (skalare Kernel-Distanzen) ---
            self.x_obs = np.array(x_obs).reshape(-1)
            self.kernel = lambda X, Y: kernel(X, Y, length_scale=length_scale, outputscale=outputscale)
        else:
            # --- Fall dim > 1: vektorisierter Kernel ueber R^d (Erweiterung aus
            # homo_bo_compare_af.ipynb, Zelle 19) ---
            if kernel not in _ND_KERNELS:
                raise ValueError(f"Kein mehrdimensionaler Kernel fuer {kernel.__name__} hinterlegt.")
            nd_kernel = _ND_KERNELS[kernel]
            self.x_obs = np.atleast_2d(np.array(x_obs))
            self.kernel = lambda X, Y: nd_kernel(X, Y, length_scale=length_scale, outputscale=outputscale)

        # C, C_inv, s_k, S_k vorab berechnen (Rest identisch fuer dim=1 und dim>1)
        self.C = self.kernel(self.x_obs, self.x_obs)
        self.C_inv = self.robust_inverse(self.C, self.x_obs)
        self.s_k = self.C_inv @ np.ones(len(self.x_obs))
        self.S_k = np.ones(len(self.x_obs)) @ self.C_inv @ np.ones(len(self.x_obs))

        # Prior-Mittelwert-Kandidaten
        rng = np.random.default_rng(42)
        self.Ms = np.sort(rng.uniform(0, 5, 250))
        self.Ms_ = np.concatenate([self.Ms, -self.Ms])

        # worst-case Prior
        self.worst_M, self.worst_h = self.compute_worst_case_prior()

        # most-likely Prior (MLL)
        self.mlls = None
        self.most_likely_M, self.most_likely_h = self.compute_most_likely_prior()

    def compute_worst_case_prior(self):
        """M und h, die den Prior am weitesten vom beobachteten Mittel entfernen."""
        worst_dist = -np.inf
        worst_M, worst_h = None, None
        y_mean = np.mean(self.y_obs)

        for M in self.Ms:
            for h in [-1, 1]:
                dist_to_obs = np.abs(M * h - y_mean)
                if dist_to_obs > worst_dist:
                    worst_dist = dist_to_obs
                    worst_M, worst_h = M, h
        return worst_M, worst_h

    def compute_most_likely_prior(self):
        """M und h mit der groessten Marginal-Log-Likelihood (MLL)."""
        mlls = [self.compute_mll_for_prior_mean(M) for M in self.Ms_]
        self.mlls = mlls
        idx = np.argmax(mlls)
        M_ml = float(np.abs(self.Ms_[idx]))
        h_ml = float(np.sign(self.Ms_[idx]))
        return M_ml, h_ml

    def k_vec(self, x):
        """k(x) als Vektor zwischen x und allen beobachteten Punkten."""
        return np.array([self.kernel(np.array([x]), np.array([xi]))[0, 0] for xi in self.x_obs])

    def k_scalar(self, x):
        """k(x,x), skalare Selbst-Kovarianz."""
        return self.kernel(np.array([x]), np.array([x]))[0, 0]

    def mu_bounds_diff(self, x):
        """Differenz zwischen den mu-Grenzen an x (Breite des Credal-Sets)."""
        kx = self.kernel(np.array([x]), self.x_obs).flatten()
        term = self.s_k @ self.y_obs / self.S_k
        if np.abs(term) > 1 + self.c / self.S_k:
            diff = (1 - kx @ self.s_k) * (term + self.c / self.S_k - term / (self.c + self.S_k))
        else:
            diff = 2 * self.c * np.abs(1 - kx @ self.s_k) / self.S_k
        return diff

    def update(self):
        """C, C_inv, s_k, S_k nach einem Daten-Update neu berechnen."""
        self.C = self.kernel(np.array(self.x_obs), np.array(self.x_obs))
        self.C_inv = self.robust_inverse(self.C, np.array(self.x_obs))
        self.s_k = self.C_inv @ np.ones(len(self.x_obs))
        self.S_k = np.ones(len(self.x_obs)) @ self.C_inv @ np.ones(len(self.x_obs))

    def robust_inverse(self, C, x_obs):
        if self.y_obs_vars is None:
            return np.linalg.inv(C + 1e-8 * np.eye(len(x_obs)))
        else:
            return np.linalg.inv(C + self.y_obs_vars * np.eye(len(x_obs)))

    def predict_posterior(self, x_new, M=0.0, h=1):
        """GP-Posterior-Mittelwert/-Varianz fuer einen neuen Punkt.

        M: Prior-Mittelwert-Parameter
        h: +/- 1
        """
        num_obs = len(self.y_obs)

        k = np.array([self.kernel(np.array([x_new]), np.array([xi]))[0, 0] for xi in self.x_obs])
        K_xx = self.kernel(np.array([x_new]), np.array([x_new]))[0, 0]

        y_hat = ((M + 1) * self.s_k.T @ self.y_obs + self.c * M * h) / (self.c + (M + 1) * self.S_k)

        mu_pred = k @ self.C_inv @ (self.y_obs - y_hat * np.ones(num_obs)) + y_hat

        prior_term = (M + 1) * (1 - k @ self.s_k) * (1 - k.T @ self.s_k) / (self.c + (M + 1) * self.S_k)
        sigma2_pred = K_xx - k @ self.C_inv @ k.T + prior_term

        return mu_pred, sigma2_pred

    def compute_mll_for_prior_mean(self, M):
        """Marginal-Log-Likelihood fuer einen gegebenen konstanten Prior-Mittelwert M."""
        y_centered = self.y_obs - M
        return -0.5 * y_centered.T @ self.C_inv @ y_centered

    def UCB(self, x, tau=1.0):
        """Upper-Confidence-Bound am worst-/most-likely-Prior (=Standardposterior, M=0).

        Vorzeichen an die Maximierungs-Konvention der Pipeline angepasst (siehe
        core_bo.py/evaluate_bbob_function): +mu statt -mu, sonst sucht die
        Akquisition faktisch das Minimum von y_obs statt des Maximums.
        """
        mu, sigma = self.predict_posterior(x)
        return mu + tau * sigma

    def PROBO(self, x, tau=1.0, rho=1.0):
        """GLCB nach Rodemann: Standard-UCB-Term + rho * Breite des Credal-Sets.

        Vorzeichen von mean_term analog zu UCB() an die Maximierungs-Konvention
        angepasst (+mu statt -mu).
        """
        kx = self.k_vec(x)
        mean_term = kx @ self.C_inv @ np.array(self.y_obs)
        var_term = tau * (self.k_scalar(x) - kx @ self.C_inv @ kx.T)
        return mean_term + var_term + rho * self.mu_bounds_diff(x)

        # Nach Definition 11 im Paper
