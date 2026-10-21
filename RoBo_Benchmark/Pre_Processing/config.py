"""Zentrale Konfiguration fuer die verteilte BO-Benchmark-Pipeline (ROBO_Benchmark).

Wird von prepare_tasks.py (Erzeugung der Task-Liste) und run_instance.py
(Ausfuehrung auf den einzelnen LRZ-Instanzen) verwendet.
"""

from pathlib import Path

# --- Verzeichnisse -------------------------------------------------------
# config.py liegt in Pre_Processing/, ROBO_Benchmark-Wurzel (mit data/) ist
# daher eine Ebene hoeher.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# --- BO-Parameter (Kreuzprodukt ergibt die Kombinationen) ----------------
# TEST-WERTE fuer einen lokalen Laufzeit-Messlauf auf diesem PC - fuer den
# echten Lauf auf dem LRZ hier wieder auf die vollen Listen hochskalieren.
FUNKTIONEN = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24] 
# [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]          # BBOB-Funktionen 1-24
DIMENSIONEN = [2, 3, 5] 
# [2, 3, 5, 10, 20, 40]                   # Suchraum-Dimensionen

# je (Surrogatmodell, Akquisitionsfunktion)-Paar eine Zeile.
# Implementiert (core_bo.py -> select_and_run_variant()), analog zu
# waehle_und_berechne_variante() in ROBO/coco_benchmark.py:
#   "GausianProzess"                    -> "EI" | "UCB" | "LCB"
#   "Imprecise_GausianProzess_Rodemann" -> "LCB" | "GLCB"   (nur dim=1, siehe imprecise_gp.py)
#   "Relevance Pursuit"                 -> "EI" | "UCB" | "LCB"
#   "STABLEOPT"                         -> nur "UCB"        (siehe stable_opt.py)
#   "DRBO"                              -> nur "distributionally robust UCB-Akquisition"
#                                          (letzte Spalte von dim = Kontextvariable, siehe drbo.py)
#   "AIRBO"                             -> nur "UCB" (siehe airbo.py)
#                                          ACHTUNG: aktuell quadratische Laufzeit
#                                          (~5s/Iteration schon bei 50 Beobachtungen) -
#                                          vor grossen Laeufen n_x_candidates/
#                                          n_cloud_samples/sub_samp_size in airbo.py
#                                          reduzieren oder Kernel vektorisieren, siehe README.md.
SURROGATE_AKQUISITION_PAARE = [
    ("GausianProzess", "EI"), # Kein LCB da hier schon -f(x) als Zielwert verwendet wird, siehe core_bo.py:run_task() und
    #("Imprecise_GausianProzess_Rodemann", "GLCB"),
    #("Relevance Pursuit", "EI"),
    #("STABLEOPT", "UCB"),
    #("DRBO", "distributionally robust UCB-Akquisition"),
    #("AIRBO", "UCB"),
]
N_SAMPLE_INIT = [5]                 # Groesse des initialen Samples
#               [2,5,10,20]

# Anzahl BO-Iterationen pro Lauf, abhaengig von der Dimension (hoehere
# Dimension -> mehr Iterationen noetig). Je Dimension eine Liste, damit bei
# Bedarf auch mehrere Iterationswerte pro Dimension getestet werden koennen.
# Muss fuer jeden Wert in DIMENSIONEN einen Eintrag enthalten.
N_ITERATION_PRO_DIMENSION = {
    2: [100],
    3: [100],
    5: [100],
    10: [100],
    20: [100],
    40: [100],
}

# --- Statistische Wiederholungen (Tensorbloecke) -------------------------
N_SAMPLE_STAT = 40               # Messlauf: klein gehalten, nur fuer Zeitmessung
SEEDS = [
    48291, 730184, 15937, 904622, 318450, 67219, 845301, 290776, 513908, 76402,
    5898, 130626, 186734, 204177, 213143, 227336, 229135, 248246, 266314, 316758,
    332814, 391109, 441839, 567503, 598308, 613474, 642683, 667237, 672755, 676254,
    699214, 706155, 732433, 788646, 797364, 839424, 914911, 941801, 948880, 988455,
]

# --- Suchraum --------------------------------------------------------------
# BBOB/COCO-Standard-Region-of-Interest: das bekannte Optimum jeder der
# 24 Funktionen liegt garantiert innerhalb von [-5, 5]^Dimension.
MINUS_AREA = -5
PLUS_AREA = 5

# --- Optimum je (Funktion, Dimension) fuer den Regret -----------------------
# f_opt (der Zielwert) kommt seit prepare_tasks.get_bbob_fopt_exact() EXAKT
# aus dem COCO-Observer, keine Schaetzung mehr noetig - siehe Erklaerung im Chat.
#
# x_opt (der zugehoerige Punkt) ist ueber cocoex dagegen nicht auslesbar und
# bleibt daher eine Zufallsstichprobe-Naeherung (prepare_tasks.estimate_bbob_argmax,
# nur fuer die Nebenmetrik x_distance relevant). Anzahl Stichproben ist
# dimensionsabhaengig (Fluch der Dimensionalitaet: bei gleicher Punktzahl wird
# die [-5,5]^d-Box mit steigendem d immer duenner abgedeckt, die Naeherung an
# x_opt also zunehmend ungenauer). Muss fuer jeden Wert in DIMENSIONEN einen
# Eintrag enthalten.
N_OPTIMUM_ESTIMATE_SAMPLES_PRO_DIMENSION = {
    2: 2000,
    3: 2000,
    5: 2000,
    10: 5000,
    20: 20000,
    40: 50000,
}
OPTIMUM_ESTIMATE_SEED = 999

# --- Fruehzeitiger Abbruch, wenn simple_regret nahe 0 ist -------------------
# Sobald |simple_regret| unter die Schwelle faellt, bricht run_task() die
# restlichen Iterationen dieser einen Task ab (der Worker-Prozess nimmt
# danach sofort die naechste Task aus der Queue) - spart Rechenzeit, wenn ein
# Verfahren die (approximierte) Optimalloesung schon erreicht hat.
#
# Die Schwelle ist bewusst RELATIV zu |f_opt_approx| (statt ein fixes
# Epsilon), damit sie auf allen Dimensionen gleich gut greift: f_opt_approx
# liegt je nach Funktion/Dimension auf ganz unterschiedlichen
# Groessenordnungen (z.B. ~-80 bei d=2, ~-1200 bei d=40, siehe
# kombinationen.csv), ein fixes Epsilon waere bei hohen Dimensionen also
# praktisch wirkungslos gewesen.
#   Schwelle = max(EARLY_STOP_MIN_EPSILON, EARLY_STOP_RELATIVE_EPSILON * |f_opt_approx|)
# EARLY_STOP_MIN_EPSILON ist nur eine Untergrenze fuer den (seltenen) Fall
# f_opt_approx nahe 0, damit die Schwelle nie auf 0 kollabiert.
EARLY_STOP_ENABLED = True
EARLY_STOP_RELATIVE_EPSILON = 0.001      # 0.1% von |f_opt_approx|
EARLY_STOP_MIN_EPSILON = 1e-6

# --- Rauschen: vier unabhaengige Schalter -----------------------------------
# Jeweils getrennt fuer X (Eingabe) und Y (Zielwert) sowie fuer die initialen
# Samples (INIT) und die waehrend BO vorgeschlagenen Punkte (ITER):
#   NOISE_X_INIT_ENABLED: Rauschen auf die initialen Sample-Positionen, einmalig
#                         in prepare_tasks.py:build_tensor_noise(). Ausgewertet
#                         und gespeichert wird die verrauschte Position.
#   NOISE_X_ITER_ENABLED: echtes Eingaberauschen pro Iteration in
#                         core_bo.run_task(): das Modell bekommt das
#                         vorgeschlagene next_x, ausgewertet wird aber
#                         next_x + eps (auf [MINUS_AREA, PLUS_AREA] geclippt).
#   NOISE_Y_INIT_ENABLED: Rauschen auf die initialen train_Y.
#   NOISE_Y_ITER_ENABLED: Rauschen auf jedes waehrend BO ausgewertete next_y.
NOISE_X_INIT_ENABLED = False
NOISE_X_ITER_ENABLED = False
NOISE_Y_INIT_ENABLED = True
NOISE_Y_ITER_ENABLED = True


def noise_config_label():
    """Menschenlesbares Label fuer die aktuelle Rauschkonfiguration - wird als
    eigene Spalte 'noise_config' in kombinationen.csv/tasks.csv/results.csv
    mitgefuehrt, damit jede Ergebniszeile fuer sich erkennen laesst, unter
    welcher Schalterkombination sie entstanden ist, statt das nur ueber den
    Ausgabe-Ordnernamen des jeweiligen Laufs zu erschliessen.
    Beispiele: "xinit_yinit_yiter", "xiter", "none"."""
    teile = [
        name for name, aktiv in (
            ("xinit", NOISE_X_INIT_ENABLED),
            ("xiter", NOISE_X_ITER_ENABLED),
            ("yinit", NOISE_Y_INIT_ENABLED),
            ("yiter", NOISE_Y_ITER_ENABLED),
            ("ctx", CONTEXT_SCENARIO_ENABLED),
        ) if aktiv
    ]
    return "_".join(teile) if teile else "none"

# X-Rauschen auf den initialen Samples (analog add_noise_stratified),
# angewendet einmalig in prepare_tasks.py:build_tensor_noise().
NOISE_ANTEIL = 1.0
NOISE_SIGMA = 0.2                   # ~2% der Suchraumbreite bei [-5,5]
NOISE_SEED_AUSWAHL = 42
NOISE_SEED_RAUSCHEN = 123

# X-Rauschen pro BO-Iteration (NOISE_X_ITER_ENABLED), angewendet in
# core_bo.run_task(). Deterministisch pro (combination_id, tensor_block)
# geseedet, damit Reruns reproduzierbar sind.
NOISE_X_ITER_ANTEIL = 1.0           # Anteil der Iterationen, deren next_x verrauscht ausgewertet wird
NOISE_X_ITER_SIGMA = 0.2            # pro Koordinate, ~2% der Suchraumbreite bei [-5,5]
NOISE_X_ITER_SEED = 555

# Y-Rauschen auf die Zielwerte (-f(x)), angewendet pro Funktionsauswertung
# in core_bo.run_task() - auf die initialen train_Y (NOISE_Y_INIT_ENABLED)
# und/oder jedes waehrend BO neu ausgewertete next_y (NOISE_Y_ITER_ENABLED).
# Deterministisch pro (combination_id, tensor_block) geseedet, damit Reruns
# reproduzierbar sind.
NOISE_Y_ANTEIL = 1.0                # Anteil der Y-Auswertungen, die Rauschen bekommen
NOISE_Y_SIGMA = 0.2                 # Rauschen auf den (negierten) Zielwert
NOISE_Y_SEED = 777

# --- Seed fuer die Zufallszahlen INNERHALB der BO-Verfahren -----------------
# (optimize_acqf-Restarts, Kandidaten-Sampling von StableOpt/DRBO/AIRBO/
# Imprecise GP). core_bo._seed_global_rngs() leitet daraus pro Task einen
# eigenen Seed ab, siehe core_bo._task_seed().
BO_SEED = 2024

# --- Kennzahlen (core_bo.compute_all_metrics) -------------------------------
# Alle Kennzahlen werden mit dem RAUSCHFREIEN f berechnet, am empfohlenen
# Punkt x_rec (beobachteter Punkt mit hoechstem Posterior-Mittelwert des
# jeweiligen Verfahrens). Die Stoerungsmengen werden einmal pro Dimension aus
# METRIC_SEED gezogen und sind fuer ALLE Verfahren/Tasks identisch - sie
# verbrauchen keine Zufallszahlen aus den Rausch-RNGs.
#
# robust_worstcase: min ueber delta in der eps-Kugel von f(x_rec + delta)
#                   (Ziel von StableOpt). METRIC_EPSILON = 0.1 entspricht dem
#                   internen eps von stable_opt.py (1% der Suchraumbreite).
# robust_expected / robust_std: Mittelwert/Std von f(x_rec + eps),
#                   eps ~ N(0, METRIC_SIGMA_X^2 I) (Ziel von AIRBO). Bewusst an
#                   das X-Rauschen der Pipeline gekoppelt, damit die Kennzahl
#                   die Guete unter dem simulierten Eingaberauschen misst.
METRIC_EPSILON = 0.1
METRIC_SIGMA_X = NOISE_X_ITER_SIGMA
METRIC_N_WORSTCASE = 200            # Stoerungen in der eps-Kugel (inkl. Mittelpunkt und Achsenpunkten auf dem Rand)
METRIC_N_EXPECTED = 500             # Monte-Carlo-Stichproben fuer robust_expected
METRIC_SEED = 4242

# --- Kontext-Szenario (fuer DRBO, Kirschner et al. 2020) ---------------------
# Wenn True, ist die LETZTE Koordinate jedes Punkts ein Kontext c, den die
# UMGEBUNG zieht (c ~ P*, gestutzte Normalverteilung), nicht das Verfahren:
# das vorgeschlagene next_x wird in der letzten Koordinate durch c_t ersetzt,
# das Modell sieht den tatsaechlichen Kontext. Gilt fuer ALLE Verfahren
# gleichermassen (fairer Vergleich) und auch fuer die initialen Samples.
# Zusaetzliche Kennzahlen (sonst NaN):
#   context_expected:  Mittelwert von f(x_rec_steuer, c) ueber eine feste
#                      Stichprobe aus P*
#   context_worstcase: schlechtester Erwartungswert ueber alle Verteilungen in
#                      der MMD-Kugel (Radius METRIC_CONTEXT_EPSILON) um diese
#                      Stichprobe - das eigentliche DRBO-Ziel
CONTEXT_SCENARIO_ENABLED = False
CONTEXT_MEAN = 0.0
CONTEXT_SIGMA = 1.5
CONTEXT_SEED = 9991
METRIC_N_CONTEXT = 50
METRIC_CONTEXT_EPSILON = 0.1
METRIC_CONTEXT_MMD_LENGTHSCALE = 1.0

# --- Zusaetzliche Ergebnis-Speicherung pro Verfahren ------------------------
# Wenn True, werden fuer die in RESULT_PER_VERFAHREN_NAMEN gelisteten
# Surrogatmodelle (surrogate_model-Wert, z.B. "GausianProzess") die Ergebnisse
# ZUSAETZLICH zur zentralen results.csv (siehe run_instance.py) in einer
# eigenen Ordnerstruktur abgelegt - ein Ordner pro Verfahren, darin eine Datei
# je (Verfahren, Akquisitionsfunktion):
#   <output_dir>/<Verfahren>/<Verfahren>_<Akquisitionsfunktion>_DIM[<alle DIMENSIONEN>]_<noise_config>_<Zeitstempel>.csv
# Die Dimensionsliste im Dateinamen ist immer die volle aktive config.DIMENSIONEN
# (nicht nur die Dimension der jeweiligen Zeile) - alle Zeilen eines Verfahrens
# landen also in derselben Datei, unabhaengig von ihrer einzelnen Dimension.
# Der Zeitstempel wird einmal pro run_instance.py-Aufruf gesetzt, damit
# innerhalb eines Laufs immer in dieselbe Datei geschrieben/angehaengt wird.
#
# Verfahren, die hier NICHT gelistet sind, laufen unveraendert weiter - nur
# ueber die zentrale results.csv (die weiterhin fuer Resume-Erkennung
# (load_done_task_ids()) sowie merge_results.py/pivot_simple_regret.py
# massgeblich ist).
RESULT_PER_VERFAHREN_ENABLED = True
RESULT_PER_VERFAHREN_NAMEN = ["GausianProzess"]   # z.B. ["GausianProzess", "Imprecise_GausianProzess_Rodemann"]

# --- AWS: Tasks zusaetzlich in eine SQS-Queue schreiben ---------------------
# Wenn True, schreibt prepare_tasks.py am Ende zusaetzlich jede Zeile aus
# tasks.csv als Nachricht in SQS_QUEUE_URL (fuer den AWS-Lauf: jede Instanz
# holt sich ihre Tasks dann von dort statt aus einer lokal zugewiesenen
# tasks.csv, siehe Main_Prozess/run_instance.py).
# Wenn False, verhaelt sich prepare_tasks.py exakt wie bisher (rein lokal,
# kein AWS-Kontakt) - z.B. fuer lokale Testlaeufe auf diesem PC.
SQS_PUSH_ENABLED = False
SQS_QUEUE_URL = ""   # z.B. "https://sqs.eu-central-1.amazonaws.com/<account-id>/robo-benchmark-tasks"
AWS_REGION = "eu-central-1"

# --- Verteilung auf die LRZ-Instanzen ---------------------------------------
# Fuer den lokalen Lauf auf diesem PC (6 physische / 12 logische Kerne):
# 1 "Instanz", 10 Worker-Prozesse (2 Kerne bleiben fuer OS/Hauptprozess frei).
# Fuer den echten LRZ-Lauf: N_INSTANZEN=4, WORKERS=10.
N_INSTANZEN = 1
WORKERS_PRO_INSTANZ = 10
