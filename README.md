# Bachelorarbeit: Vergleich robuster Bayesianischer Optimierungsverfahren

Diese Bachelorarbeit beschäftigt sich mit dem Vergleich verschiedener
Bayes'scher Optimierungsverfahren (BO) – insbesondere robuster Varianten, die
mit Unsicherheit in den Eingabedaten, im Modell oder in der Zielfunktion
umgehen können – gegen einen klassischen Gaußprozess als Baseline. Als
Testumgebung dient die BBOB/COCO-Benchmarksuite. Ziel ist ein Dataframe, der
für jede getestete Parameterkombination die Messwerte je Iteration speichert
und sich damit über gängige Analyse-Pipelines (pandas) auswerten lässt.

## Zielsetzung

Welche BO-Variante konvergiert unter welchen Bedingungen (Zielfunktion,
Dimension, Rauschen, Startstichprobe) schneller bzw. robuster als der
Standard-Gaußprozess? Dazu werden alle Verfahren mit identischen
Startbedingungen (gleiche Stichproben, gleiche Seeds) auf denselben
Benchmarkfunktionen ausgeführt und ihre Konvergenzgeschwindigkeit (Simple
Regret über die Iteration) verglichen.

## Input-Parameter

| Parameter | Beschreibung | Status |
|---|---|---|
| Surrogatmodell | welches BO-Verfahren (siehe Tabelle unten) | nicht vollständig |
| Akquisitionsfunktion | z. B. EI, UCB, LCB (verfahrensabhängig) | implementiert |
| Anzahl Dimensionen | Suchraumdimension der Zielfunktion | implementiert |
| Zielfunktion | BBOB-Funktion 1–24 | implementiert |
| Stichprobengröße | Größe der Initialstichprobe | implementiert |
| Anzahl Iterationen | Länge des BO-Laufs | implementiert |
| Metriken für Funktionsähnlichkeit | Lp-Distanzen, MSE, Cosine Similarity, Wasserstein-Distanz, Bregman-Divergenz | **konzeptionell/geplant** – bislang nur ein einzelner Lp-Abstand (`p=2`) zum bekannten Optimum als Prototyp in `ROBO/Zusatz Funktioen .py`, noch nicht in der produktiven Pipeline |

## Workflow

1. **Eingabe** der zu testenden Zielfunktionen, Dimensionen, Surrogatmodelle,
   Akquisitionsfunktionen, Stichprobengrößen und Iterationszahlen –
   zentral in `config.py`.
2. **Kombinationen erzeugen**: Kreuzprodukt aus Zielfunktion, Dimension,
   (Surrogatmodell, Akquisitionsfunktion), Stichprobengröße und
   Iterationszahl ergibt die Parameterkombinationen (`kombinationen.csv`).
   Jede Kombination wird zusätzlich mit mehreren unabhängigen
   Stichproben-Tensorblöcken (statistische Wiederholungen) verknüpft → Tasks
   (`tasks.csv`).
3. **Schleife über die Tasks**: jede Task = eine (Kombination ×
   Tensorblock)-Kombination.
4. **Testdaten erzeugen**: Aufruf der `bbob`-Suite aus dem `cocoex`-Package
   (COCO-Plattform) für die jeweilige Zielfunktion/Dimension/Instanz.
5. **BO-Lauf**: für jede Task werden pro Iteration Surrogatmodell gefittet,
   nächster Punkt vorgeschlagen, ausgewertet und die Regret-Metriken
   berechnet (`best_so_far`, `simple_regret`, `immediate_regret`,
   `x_distance`) – eine Ergebniszeile je Iteration.
6. **Zusammenführen**: alle Ergebnis-Dataframes (parallel über mehrere
   LRZ-Instanzen berechnet) werden zu einer Gesamtdatei zusammengeführt.
7. **Analysen**: Pivotierung ins Wide-Format je Kombination sowie Mittelwert
   und Konfidenzintervall über die statistischen Wiederholungen.
8. **Plots**: Konvergenzgeschwindigkeit (Mittelwert ± Konfidenzband über die
   Iteration), gefiltert und vergleichbar über mehrere BO-Varianten hinweg.

Die produktive Umsetzung dieses Workflows liegt in [`ROBO_Benchmark/`](ROBO_Benchmark/)
(siehe [`ROBO_Benchmark/README.md`](ROBO_Benchmark/README.md) für Details zu
Konfiguration, Ausführung und Ergebnisformat). Frühere/explorative Skripte
liegen in [`ROBO/`](ROBO/).

## Implementierte BO-Verfahren

| Verfahren | Kernidee | Referenz |
|---|---|---|
| Standard-Gaußprozess | klassische BO als Baseline (EI/UCB/LCB) | [BoTorch] |
| Imprecise Bayesian Optimization (PROBO) | Menge unsicherer Priori-Mittelwerte (Credal Set) statt einem festen Prior | [Rodemann & Augustin 2024] |
| Relevance Pursuit | robuste, datenpunktspezifische Rauschvarianz zur automatischen Ausreißererkennung | [Ament et al. 2024] |
| STABLEOPT | robustes Optimum über min-max UCB/LCB in einer ε-Kugel | [Bogunovic et al. 2018] |
| DRBO (Distributionally Robust BO) | MMD-basierte Worst-Case-Gewichtung über Kontextverteilungen | [Kirschner et al. 2020] |
| AIRBO | MMD-Kernel über Eingabeunsicherheits-Stichprobenwolken + Nyström-Approximation | [Yang et al. 2023] |

Details, Einschränkungen (z. B. Dimensionsgrenzen, Laufzeit) und die genaue
Code-Zuordnung stehen in [`ROBO_Benchmark/README.md`](ROBO_Benchmark/README.md).

## Visualisierungen

Zwei begleitende, interaktive Schaubilder zum Projekt:

- **[Architektur & Pipeline von ROBO_Benchmark](https://claude.ai/code/artifact/d445f4b0-bb66-49de-8a24-7106362adde2)**
  – Ordnerstruktur, verfügbare Parameter, die komplette Pipeline vom
  Task-Erzeugen bis zum Plot, Lokal/LRZ-Deployment und eine
  Schritt-für-Schritt-Anleitung mit den tatsächlichen Befehlen.
- **[Ablauf der BO-Iterationsschleife (`run_task()`)](https://claude.ai/code/artifact/adc072d4-31ab-4ad3-8d8b-633f7ade1085)**
  – der eigentliche Programmablauf: Initialstichprobe, Varianten-Dispatch an
  die sechs BO-Verfahren, Regret-Berechnung, mit Funktionsreferenz.

> Hinweis: Beide Seiten sind zunächst privat. Falls dieses README öffentlich
> (z. B. auf GitHub) einsehbar sein soll, müssen die Links vorher über das
> Teilen-Menü der jeweiligen Seite freigegeben werden, sonst funktionieren
> sie nur für dich.

## Literatur

1. Garnett, R. (2023). *Bayesian Optimization*. Cambridge University Press.
   [cambridge.org](https://www.cambridge.org/core/books/bayesian-optimization/11AED383B208E7F22A4CE1B5BCBADB44)
2. Frazier, P. I. (2018). *A Tutorial on Bayesian Optimization*. arXiv:1807.02811.
   [arxiv.org/abs/1807.02811](https://arxiv.org/abs/1807.02811)
3. Balandat, M. et al. (2020). *BoTorch: A Framework for Efficient Monte-Carlo
   Bayesian Optimization*. [botorch.org](https://botorch.org/)
4. Rodemann, J., & Augustin, T. (2024). *Imprecise Bayesian Optimization*.
   Knowledge-Based Systems, 300, 112186.
   [sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0950705124008207)
5. Bogunovic, I., Scarlett, J., Jegelka, S., & Cevher, V. (2018).
   *Adversarially Robust Optimization with Gaussian Processes*. NeurIPS 2018,
   arXiv:1810.10775. [arxiv.org/abs/1810.10775](https://arxiv.org/abs/1810.10775)
6. Kirschner, J., Bogunovic, I., Jegelka, S., & Krause, A. (2020).
   *Distributionally Robust Bayesian Optimization*. AISTATS 2020,
   arXiv:2002.09038. [arxiv.org/abs/2002.09038](https://arxiv.org/abs/2002.09038)
7. Ament, S., Santorella, E., Eriksson, D., Letham, B., Balandat, M., &
   Bakshy, E. (2024). *Robust Gaussian Processes via Relevance Pursuit*.
   NeurIPS 2024, arXiv:2410.24222.
   [arxiv.org/abs/2410.24222](https://arxiv.org/abs/2410.24222)
8. Yang, L., Lyu, J., Lyu, W., & Chen, Z. (2023). *Efficient Robust Bayesian
   Optimization for Arbitrary Uncertain Inputs* (AIRBO). NeurIPS 2023,
   arXiv:2310.20145. [arxiv.org/abs/2310.20145](https://arxiv.org/abs/2310.20145) ·
   Referenzimplementierung: [github.com/huawei-noah/HEBO](https://github.com/huawei-noah/HEBO)
9. Fohler, L. (2025). *A comparison between Bayesian optimization frameworks
   and reinforcement learning for dose optimization*. Masterarbeit,
   Fraunhofer-Institut.
   [publica.fraunhofer.de](https://publica.fraunhofer.de/entities/publication/8eea9127-4f42-4cb7-b6ef-4274e9d68ad9)
10. Hansen, N. et al. *COCO: A platform for comparing continuous optimizers
    in a black-box setting* – BBOB-Testsuite.
    [coco-platform.org/testsuites/bbob](https://coco-platform.org/testsuites/bbob/overview.html) ·
    [cocoex API-Dokumentation](https://numbbo.github.io/coco-doc/apidocs/cocoex/) ·
    [github.com/numbbo/coco](https://github.com/numbbo/coco)

## Repository-Struktur (Kurzüberblick)

```
Bachelorarbeit/
├── ROBO_Benchmark/     produktive, dokumentierte Benchmark-Pipeline (siehe eigenes README)
├── ROBO/               frühere/explorative Skripte und Prototypen
├── ROBO_NEU/           Zwischenstand vor der Umstrukturierung zu ROBO_Benchmark
├── Literatur/          Paper-PDFs zu den robusten BO-Verfahren
└── tcs_thesis_template/ LaTeX-Vorlage der Bachelorarbeit
```
