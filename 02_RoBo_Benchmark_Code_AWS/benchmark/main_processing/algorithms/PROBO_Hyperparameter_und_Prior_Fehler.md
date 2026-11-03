# PROBO: Kernel-Hyperparameter, Prior-Fehler und Wahl von c

## Kurzfassung
- PROBO fittet **Länge und Outputscale** des Kernels in jeder Iteration per
  Marginal Likelihood, wie GP_BO, DRBO, StableOpt und RP_BO (früher fest 1.0 / 1.0).
- Im Szenario `prior_error` geschieht das unter dem **eingefrorenen falschen
  Prior-Mittelwert δ**, wie bei den anderen Verfahren.
- Der Impräzisionsparameter ist **c = 1** (früher 50). Bei c = 50 dominiert die
  Breite des Credal-Sets die GLCB so stark, dass sie fast reine Exploration ist und
  einen Prior-Fehler δ überdeckt.

Code: `probo.py` → `fit_kernel_hyperparameters()`, `run_imprecise_gausian_prozess()`;
Einstellungen in `preprocessing/config.py` → `ALGORITHM_CONFIGS["PROBO"]`
(`fit_hyperparameters`, `c`).

## 1. Fit der Kernel-Hyperparameter
`fit_kernel_hyperparameters()` baut ein BoTorch-`SingleTaskGP`, das zu
`BayesOptimizer` (`uncertainty_aware_bo.py`, unverändert) passt:

| Baustein | Wahl | Grund |
|---|---|---|
| Kernel | isotrop (eine Länge), Matern-3/2 oder RBF, mit `ScaleKernel` | dieselben Formeln wie `matern()` / `rbf()` |
| y | bereits standardisiert, kein `Standardize` | PROBO standardisiert selbst |
| Rauschen | fest 1e-6 | `BayesOptimizer` rechnet rauschfrei (Jitter 1e-8) |
| Prior-Mittelwert | eingefroren auf δ (nominal: 0) | wie `apply_prior_error` bei den anderen Verfahren |
| Prior auf der Länge | dimensionsskalierter LogNormal-Prior (Hvarfner et al. 2024) | wie GP_BO |

Die Eingaben werden für den Fit auf [0, 1] normalisiert, die Länge danach in den
Suchraum zurückgerechnet: `length_scale = l_norm * (upper - lower)`.

Mit `"fit_hyperparameters": False` gelten wieder die festen Werte `length_scale`
und `outputscale` aus der Config.

## 2. Wie der Prior-Fehler in PROBO wirkt
PROBO hat kein Mittelwert-Modul. Es nutzt die Identität: Ein GP mit
Prior-Mittelwert δ auf ỹ ist identisch zu einem GP mit Prior-Mittelwert 0 auf
ỹ − δ. Ablauf in `run_imprecise_gausian_prozess()`:

1. y standardisieren: ỹ = (y − mean) / std (neu in jeder Iteration)
2. Hyperparameter fitten (auf ỹ, Mittelwert δ eingefroren)
3. Daten verschieben: ỹ − δ
4. Vorzeichen drehen (−y), weil `BayesOptimizer` maximiert
5. In `posterior_mean` wird δ wieder addiert und destandardisiert

Die GLCB (`PROBO_batch`) besteht aus drei Teilen:

```
GLCB(x) ≈ μ̂_0(x) + δ·w(x)  −  τ·Var(x)  −  ρ·(2c/S_k)·|w(x)|
          └ Mittelwert ┘      └ Varianz ┘   └ Breite des Credal-Sets ┘
```

mit w(x) = 1 − k(x)ᵀC⁻¹**1** und S_k = **1**ᵀC⁻¹**1**.

- **Mittelwert-Term:** Er kommt von *einem* präzisen GP mit Prior-Mittelwert δ.
  Er verschiebt sich also genauso um δ·w(x) wie bei GP_BO.
- **Varianz:** Sie hängt nicht von δ ab, höchstens indirekt über die gefitteten
  Hyperparameter.
- **Breite** (`mu_bounds_diff_batch`): Solange |s_kᵀ(ỹ − δ)/S_k| ≤ 1 + c/S_k gilt,
  ist sie 2c·|w(x)|/S_k und **unabhängig von δ**. Sonst hängt auch sie von δ ab.

PROBO ist also *nicht* dadurch robust, dass es alle Prior-Mittelwerte betrachtet.
Ob δ durchschlägt, entscheidet das Verhältnis von δ zu 2c/S_k: Beide Terme hängen
am selben w(x).

## 3. Wahl von c = 1
Grob nachgerechnet mit gefitteten Hyperparametern auf einer Quadratfunktion
(also nur ein Anhaltspunkt, kein BBOB-Ergebnis):

| dim | n | 2c/S_k bei c = 50 | bei c = 1 |
|---|---|---|---|
| 2 | 10 | ~100 | ~2 |
| 5 | 20 | ~48 | ~1 |

Auf standardisierten y liegen Mittelwert (±1 bis 2) und Varianz (0 bis Outputscale)
in der Größenordnung 1. Bei **c = 1** liegt die Breite am Anfang des Laufs in
derselben Größenordnung, und keiner der Terme dominiert. Bei **c = 50** ist sie
50- bis 100-mal größer. GLCB sucht dann fast nur dort, wo w(x) groß ist, und ein
Prior-Fehler δ ≤ 1 geht darin unter.

c wurde **vorab** über die Skalen der Terme festgelegt und nicht anhand der
Ergebnisse im Szenario `prior_error`. Es gilt für alle Szenarien gleich.

Ein kurzer Test (Quadratfunktion, 12 Punkte) zeigt den erwarteten Wechsel: Mit
c = 1 schlägt PROBO Punkte nahe am Optimum vor statt am Rand des Suchraums, und ein
optimistischer Prior-Fehler (δ = −1) treibt es wieder an den Rand. Der
Prior-Fehler ist damit für PROBO messbar.

## 4. Einschränkungen
- **Kein c bleibt über den ganzen Lauf ausgewogen.** S_k ändert sich mit den Daten.
  Bei vielen, eng korrelierten Punkten wird 2c/S_k sehr groß (bei c = 50 bis
  10⁴ und mehr), während w(x) nahe an Daten klein wird.
- **Bisher nicht geprüft:** ob c = 1 im Szenario `nominal` auch besser *optimiert*
  als c = 50. Empfohlen ist ein kurzer Kontrolllauf mit c ∈ {1, 10, 50}, nur in
  `nominal`.
- c ist jetzt relativ zu einem **gefitteten** Outputscale. Die Wirkung von c hängt
  deshalb etwas von der Funktion ab.
