
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pfade  # noqa: F401 - fuegt Pre_Processing/ (config) zu sys.path hinzu

import torch
try:
    # aktueller Modulpfad (botorch >= ~0.13)
    from botorch.models.robust_relevance_pursuit_model import RobustRelevancePursuitSingleTaskGP
except ImportError:
    # aelterer Modulpfad (botorch ~0.11-0.12), zur Sicherheit als Fallback
    from botorch.models.relevance_pursuit_model import RobustRelevancePursuitSingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

import config


def run_relevance_pursuit(train_X, train_Y, dim, acquisition_func,
                           lower_bound=config.MINUS_AREA, upper_bound=config.PLUS_AREA):
    """
    Fuehrt die robuste Relevance-Pursuit-GP-Variante fuer eine Iteration aus:
    1) RobustRelevancePursuitSingleTaskGP fitten (erkennt Ausreisser automatisch
       ueber eine datenpunktspezifische Rauschvarianz, siehe Ament et al. 2024)
    2) passende Akquisitionsfunktion anhand von acquisition_func waehlen ("EI", "UCB", "LCB")
    3) naechsten Punkt per optimize_acqf vorschlagen

    train_X : torch.double, shape [n, dim]
    train_Y : torch.double, shape [n, 1] (Maximierungs-Konvention, siehe core_bo.py)
    dim     : int, Dimension des Suchraums

    Rueckgabe: next_x (torch.double, shape [dim]), acq_value (torch.double, Skalar)
    """
    bounds = torch.tensor(
        [[lower_bound] * dim, [upper_bound] * dim], dtype=torch.double
    )

    # 1) Modell fitten (Inputs auf [0,1]^d normalisiert, Outputs standardisiert,
    #    da der Suchraum bzw. die Funktionswerte sonst zu gross fuer ein stabiles Fitting sind)
    model = RobustRelevancePursuitSingleTaskGP(
        train_X, train_Y,
        input_transform=Normalize(d=dim, bounds=bounds),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # 2) Akquisitionsfunktion abfragen und passend waehlen
    if acquisition_func == "EI":
        acqf = LogExpectedImprovement(model, best_f=train_Y.max())
    elif acquisition_func == "UCB":
        acqf = UpperConfidenceBound(model, beta=2.0, maximize=True)
    elif acquisition_func == "LCB":
        acqf = UpperConfidenceBound(model, beta=2.0, maximize=False)
    else:
        raise ValueError(f"Unbekannte Akquisitionsfunktion fuer Relevance Pursuit: {acquisition_func}")

    # 3) naechsten Punkt vorschlagen
    next_x, acq_value = optimize_acqf(
        acqf,
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
    )

    return next_x.squeeze(0), acq_value