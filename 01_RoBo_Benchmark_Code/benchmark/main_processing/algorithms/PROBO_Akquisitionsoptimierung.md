# PROBO: Optimierung der Akquisitionsfunktion

## Kurzfassung
PROBO maximiert seine Akquisitionsfunktion (GLCB bzw. LCB) per **Focus Search**
mit **1000 Punkten pro Runde, 5 Runden und 5 Restarts**. Das entspricht dem
Optimierer, den Rodemann & Augustin (2024) in ihren Experimenten verwenden.

Code: `probo.py` → `focus_search()`, aufgerufen in `run_imprecise_gausian_prozess()`.

## Quelle
Rodemann, J. & Augustin, T. (2024). *Imprecise Bayesian optimization*.
Knowledge-Based Systems 300, 112186.

- Abschnitt 2: Akquisitionsfunktionen sind billig auszuwerten. Ihr Optimum lässt
  sich daher mit einfachen Verfahren wie Grid Search, Random Search oder Focus
  Search finden.
- Abschnitt 6 (Experimente): Focus Search mit 1000 Auswertungen pro Runde und
  maximal 5 Restarts (gerechnet in R mit mlrMBO).

Focus Search stammt aus mlrMBO: Bischl et al. (2017), *mlrMBO: A modular framework
for model-based optimization of expensive black-box functions*, arXiv:1703.03373, S. 7.

## Algorithmus
Nachbau von `mlrMBO::optimizeFocusSearch`:

```
für r = 1 … restarts (5):
    Box = gesamter Suchraum [lower, upper]^d
    für i = 1 … maxit (5):
        ziehe points (1000) Punkte per Latin Hypercube in der Box
        werte die Akquisitionsfunktion aus
        merke den global besten Punkt
        verkleinere die Box um den besten Punkt dieser Runde:
            pro Dimension  lo = max(lo, x* − Breite/4),  hi = min(hi, x* + Breite/4)
gib den global besten Punkt zurück
```

In jeder Runde halbiert sich die Box pro Dimension, liegt aber immer um den
bisher besten Punkt. Die Restarts beginnen wieder beim vollen Suchraum. So
verhindern sie, dass die Suche in einem lokalen Optimum hängen bleibt.

**Aufwand:** 5 × 5 × 1000 = 25.000 Auswertungen der Akquisitionsfunktion pro
BO-Iteration.

| Parameter (`run_imprecise_gausian_prozess`) | Wert | mlrMBO-Name |
|---|---|---|
| `focus_points` | 1000 | `opt.focussearch.points` |
| `focus_maxit` | 5 (mlrMBO-Standard) | `opt.focussearch.maxit` |
| `focus_restarts` | 5 (laut Paper) | `opt.restarts` |

Das Paper nennt die Rundenzahl `maxit` nicht. Deshalb gilt hier der
mlrMBO-Standardwert 5.

## Umsetzungsdetails
- **Gleich für alle Dimensionen:** Focus Search wird für jede Dimension verwendet.
  Das frühere 1D-Gitter (`linspace`, 500 Punkte) und die nD-Zufallssuche
  (500 Punkte) gibt es nicht mehr.
- **Reproduzierbarkeit:** Das LHS-Design wird aus dem `rng` gezogen, den der
  Runner pro Wiederholung übergibt. Gleicher Seed ergibt gleiches `next_x`.
- **Vektorisierung:** Damit 25.000 Auswertungen schnell genug laufen, gibt es in
  `uncertainty_aware_bo.py` die Methoden `UCB_batch`, `PROBO_batch` und
  `mu_bounds_diff_batch`. Sie rechnen dieselben Formeln wie `UCB`, `PROBO` und
  `mu_bounds_diff`, aber für alle Punkte auf einmal. Getestet ist die
  Übereinstimmung mit den skalaren Versionen bis auf ≤ 1e-10. Eine Iteration
  dauert etwa 0,1 s, auch bei d = 10.
- **Protokollierung:** In `hyperparameters` stehen jetzt `af_optimizer`,
  `focus_points`, `focus_maxit` und `focus_restarts` (vorher `n_candidates`).

## Unterschiede zum Original
- **Sprache:** Rodemann hat in R mit mlrMBO gerechnet, hier ist es ein Nachbau in
  Python. LHS kommt aus `scipy.stats.qmc.LatinHypercube`.
- **Gleichstände:** Bei gleich guten Punkten wählt mlrMBO zufällig, hier wird der
  erste genommen (`argmax`).
- **Dimension:** Das Paper betrachtet nur 1D. Für d > 1 wird dieselbe Focus Search
  ohne Änderung auf [lower, upper]^d angewendet. Die Punktzahl wird *nicht* mit
  d skaliert. Die Abdeckung des Suchraums wird mit steigender Dimension also
  gröber, die Box-Verkleinerung gleicht das teilweise aus.

## Bewusste Abweichung: Varianz statt Standardabweichung
Laut Definition 11 im Paper lautet der GLCB

    GLCB(x) = μ̂(x) − τ · √Var(μ̂(x)) − ρ · (μ̄(x) − μ̲(x))

Die Implementierung nutzt dagegen die **Varianz Var(μ̂(x)) ohne Wurzel**. Das
betrifft `PROBO`, `UCB` und die `_batch`-Versionen in `uncertainty_aware_bo.py`.
Die Vorzeichen sind an die Maximierungs-Konvention der Pipeline angepasst.

**Begründung für das Beibehalten:** So hat Rodemann es selbst in seinem
Python-Code umgesetzt (`homo_bo_compare_af.ipynb`, GLCB und LCB jeweils mit
`tau * var`). Die Implementierung folgt also seinem Code und nicht der
Formel im Paper.

**Auswirkung:** Bei standardisierten y und `outputscale = 1` liegt die
Posterior-Varianz (praktisch) in [0, 1], dort gilt Var ≤ √Var. Der Explorationsterm ist
damit schwächer als im Paper, vor allem in Bereichen mit kleiner Unsicherheit.
Bei gleichem τ exploriert das Verfahren also weniger, und die Credal-Set-Breite
(ρ-Term) bekommt relativ mehr Gewicht.
