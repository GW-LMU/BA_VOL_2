import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pfade  # noqa: F401 - fuegt Main_Prozess/, BO_Verfahren/, Post_Processing/ zu sys.path hinzu

import argparse
import glob
import itertools
import json
import re
import shutil
import uuid

import cocoex
import numpy as np
import pandas as pd

import config
import core_bo


def build_kombinationen():# ✅
    rows = []
    combination_id = 0
    for funktion, dimension, (surrogate, akquisition), n_init in itertools.product(
        config.FUNKTIONEN,
        config.DIMENSIONEN,
        config.SURROGATE_AKQUISITION_PAARE,
        config.N_SAMPLE_INIT,
    ):
        # Iterationswerte sind dimensionsabhaengig (siehe config.N_ITERATION_PRO_DIMENSION)
        for n_iter in config.N_ITERATION_PRO_DIMENSION[dimension]:
            rows.append({
                "combination_id": combination_id,
                "funktion": funktion,
                "dimension": dimension,
                "surrogate_model": surrogate,
                "acquisition_model": akquisition,
                "sample_size_initial": n_init,
                "max_iteration": n_iter,
                # konstant fuer den gesamten Lauf (siehe die NOISE_*_ENABLED-
                # Schalter in config.py) - als eigene Spalte mitgefuehrt, damit jede
                # Ergebniszeile selbst erkennen laesst, unter welchem
                # Rauschzustand sie entstanden ist.
                "noise_config": config.noise_config_label(),
            })
            combination_id += 1
    return pd.DataFrame(rows)


def estimate_bbob_argmax(funktion, dimension, suite, lower_bound, upper_bound,
                          n_samples, seed, instance=1):# ✅
    """Schaetzt NUR den Punkt x_opt, an dem -f(x) im Suchraum (ungefaehr)
    maximal wird, per Zufallsstichprobe. Wird ausschliesslich fuer die
    Nebenmetrik x_distance gebraucht (siehe core_bo.compute_regret()) - der
    zugehoerige Funktionswert f_opt kommt NICHT mehr von hier, sondern exakt
    aus get_bbob_fopt_exact(), da eine Zufallsstichprobe den wahren
    Maximalwert besonders bei hohen Dimensionen und stark zerkluefteten
    Funktionen (z.B. BBOB-Funktion 24) deutlich unterschaetzt (siehe
    Vergleich im Chat: bei d=20 teils >200% Abweichung vom wahren Wert). Der
    zugehoerige x_opt bleibt aus demselben Grund zwangslaeufig ebenfalls nur
    eine grobe Naeherung - x_distance ist entsprechend nur zur groben
    Orientierung zu verwenden, kein belastbares Genauigkeitsmass. Wird
    EINMAL pro (Funktion, Dimension)-Paar zentral hier aufgerufen, nicht pro
    Task, um die Kosten bei sehr vielen Kombinationen gering zu halten.
    """
    problem = suite.get_problem_by_function_dimension_instance(
        function=funktion, dimension=dimension, instance=instance
    )
    rng = np.random.default_rng(seed)
    candidates = rng.uniform(lower_bound, upper_bound, size=(n_samples, dimension))
    values = np.array([-problem(x) for x in candidates])
    best_idx = int(np.argmax(values))
    return candidates[best_idx]


def get_bbob_fopt_exact(funktion, dimension, suite, instance=1):# ✅ 
    """Liefert das EXAKTE f_opt (Maximum von -f(x)) direkt aus dem COCO-
    Observer/Logger - keine Schaetzung noetig.

    COCO kennt fuer jede BBOB-Instanz intern den wahren Minimalwert von f(x)
    (Rohkonvention, Precision 1e-8), gibt ihn ueber die normale Problem-API
    aber nicht heraus (Sinn eines Black-Box-Benchmarks). Haengt man jedoch
    einen 'bbob'-Observer an und wertet die Funktion EINMAL beliebig aus
    (z.B. am Nullpunkt), schreibt der Logger eine .dat-Datei, deren erste
    Zeile den Wert als Kommentar enthaelt: "... - Fopt (7.948...e+01) ...".
    Das wird hier ausgelesen und negiert (-Fopt), um dieselbe -f(x)-
    Vorzeichenkonvention wie core_bo.evaluate_bbob_function() zu erhalten.

    Der Log-Ordner wird in einen eindeutig benannten Unterordner unter
    ./exdata/ geschrieben und danach wieder geloescht, damit nichts vom
    Repo-Arbeitsverzeichnis uebrig bleibt (auch bei mehreren parallelen
    Aufrufen unkritisch, da jeder Aufruf einen eigenen Unterordner bekommt).
    """
    problem = suite.get_problem_by_function_dimension_instance(
        function=funktion, dimension=dimension, instance=instance
    )
    result_folder = f"fopt_probe_{funktion}_{dimension}_{instance}_{uuid.uuid4().hex[:8]}"
    observer = cocoex.Observer("bbob", f"result_folder: {result_folder}")
    try:
        problem.observe_with(observer)
        problem(np.zeros(dimension))  # ein einziger Dummy-Aufruf reicht, um das Log zu erzeugen
    finally:
        problem.free()

    dat_files = glob.glob(f"exdata/{result_folder}/data_f{funktion}/*.dat")
    if not dat_files:
        raise RuntimeError(
            f"COCO-Observer hat keine .dat-Datei fuer Funktion {funktion}, "
            f"Dimension {dimension} erzeugt (erwartet unter exdata/{result_folder}/)."
        )
    first_line = open(dat_files[0]).readline()
    match = re.search(r"Fopt \(([-\d.eE+]+)\)", first_line)
    if match is None:
        raise RuntimeError(
            f"Konnte 'Fopt (...)' nicht aus der .dat-Kopfzeile lesen: {first_line!r}"
        )
    fopt_raw = float(match.group(1))

    shutil.rmtree(f"exdata/{result_folder}", ignore_errors=True)

    return -fopt_raw


def add_optimum_estimates(df_gesamt):# ✅
    """Traegt je (Funktion, Dimension) EINMAL das Optimum in df_gesamt ein,
    statt es (teuer) pro Task neu zu berechnen:

    - f_opt ist EXAKT (kommt direkt aus dem COCO-Observer, siehe
      get_bbob_fopt_exact() - keine Schaetzung mehr noetig).
    - x_opt bleibt eine grobe Zufallsstichprobe-Naeherung (siehe
      estimate_bbob_argmax()), da der zugehoerige x-Punkt ueber cocoex
      nicht auslesbar ist. Nur fuer die Nebenmetrik x_distance relevant,
      nicht fuer simple_regret/immediate_regret.
    """
    suite = cocoex.Suite("bbob", "", "")

    lookup = {}
    for funktion, dimension in df_gesamt[["funktion", "dimension"]].drop_duplicates().itertuples(index=False):
        f_opt = get_bbob_fopt_exact(funktion, dimension, suite)
        x_opt = estimate_bbob_argmax(
            funktion, dimension, suite,
            config.MINUS_AREA, config.PLUS_AREA,
            n_samples=config.N_OPTIMUM_ESTIMATE_SAMPLES_PRO_DIMENSION[dimension],
            seed=config.OPTIMUM_ESTIMATE_SEED,
        )
        lookup[(funktion, dimension)] = (json.dumps(x_opt.tolist()), f_opt)
        print(f"  Funktion {funktion}, Dimension {dimension}: f_opt (exakt) = {f_opt:.4f}")

    # Spaltennamen bewusst beibehalten (x_opt_approx/f_opt_approx), obwohl
    # f_opt_approx jetzt exakt ist - core_bo.run_task() und die Tasks-CSV
    # erwarten diese Namen; ein Rename wuerde mehrere Dateien beruehren, ohne
    # funktionalen Nutzen.
    df_gesamt["x_opt_approx"] = df_gesamt.apply(
        lambda r: lookup[(r["funktion"], r["dimension"])][0], axis=1
    )
    df_gesamt["f_opt_approx"] = df_gesamt.apply(
        lambda r: lookup[(r["funktion"], r["dimension"])][1], axis=1
    )
    return df_gesamt


def build_tensor_noise(df_gesamt):# ✅
    max_dim = int(df_gesamt["dimension"].max())
    max_samples = int(df_gesamt["sample_size_initial"].max())

    tensor = np.empty((config.N_SAMPLE_STAT, max_samples, max_dim), dtype=np.float32)
    for i in range(config.N_SAMPLE_STAT):
        rng = np.random.default_rng(config.SEEDS[i])
        tensor[i] = rng.uniform(
            low=config.MINUS_AREA, high=config.PLUS_AREA,
            size=(max_samples, max_dim),
        ).astype(np.float32)

    ergebnis = tensor.copy()
    if not config.NOISE_X_INIT_ENABLED:
        return ergebnis

    # Rauschen hinzufuegen, analog add_noise_stratified() in ROBO/Subprocess.py
    rng_auswahl = np.random.default_rng(config.NOISE_SEED_AUSWAHL)
    rng_rauschen = np.random.default_rng(config.NOISE_SEED_RAUSCHEN)
    for i in range(tensor.shape[0]):
        schicht = ergebnis[i]
        n_total = schicht.size
        n_auswahl = int(round(n_total * config.NOISE_ANTEIL))
        flat_indices = rng_auswahl.choice(n_total, size=n_auswahl, replace=False)
        idx = np.unravel_index(flat_indices, schicht.shape)
        rauschen = rng_rauschen.normal(loc=0.0, scale=config.NOISE_SIGMA, size=n_auswahl)
        schicht[idx] = schicht[idx] + rauschen

    return ergebnis


def build_tasks(df_gesamt):# ✅
    rows = []
    task_id = 0
    for combo in df_gesamt.itertuples():
        for tensor_block in range(config.N_SAMPLE_STAT):
            rows.append({
                "task_id": task_id,
                "combination_id": combo.combination_id,
                "tensor_block": tensor_block,
                "funktion": combo.funktion,
                "dimension": combo.dimension,
                "surrogate_model": combo.surrogate_model,
                "acquisition_model": combo.acquisition_model,
                "sample_size_initial": combo.sample_size_initial,
                "max_iteration": combo.max_iteration,
                "noise_config": combo.noise_config,
                "x_opt_approx": combo.x_opt_approx,
                "f_opt_approx": combo.f_opt_approx,
            })
            task_id += 1
    return pd.DataFrame(rows)


def push_tasks_to_sqs(df_tasks, queue_url, region=None):
    """Schreibt jede Zeile aus df_tasks als eigene Nachricht in die SQS-Queue
    (fuer den AWS-Lauf: jede Instanz holt sich ihre Tasks von dort statt aus
    einer lokal zugewiesenen tasks.csv, siehe Main_Prozess/run_instance.py).

    Nutzt send_message_batch (max. 10 Nachrichten/Aufruf) statt einzelner
    send_message-Aufrufe, um bei 500.000+ Tasks nicht 500.000 einzelne
    API-Requests zu brauchen.
    """
    import boto3  # lokaler Import: boto3 nur noetig, wenn dieser Pfad genutzt wird

    sqs = boto3.client("sqs", region_name=region)
    records = df_tasks.to_dict("records")
    total = len(records)

    for start in range(0, total, 10):
        chunk = records[start:start + 10]
        entries = [{"Id": str(i), "MessageBody": json.dumps(row)} for i, row in enumerate(chunk)]
        response = sqs.send_message_batch(QueueUrl=queue_url, Entries=entries)
        failed = response.get("Failed")
        if failed:
            raise RuntimeError(f"SQS send_message_batch: {len(failed)} Nachrichten fehlgeschlagen: {failed}")

    print(f"{total} Tasks in SQS-Queue geschrieben: {queue_url}")


def main(sqs_queue_url=None, aws_region=None, sqs_push_enabled=None):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    df_gesamt = build_kombinationen()
    print("Schaetze Maximum je (Funktion, Dimension)...")
    df_gesamt = add_optimum_estimates(df_gesamt)
    tensor_noise = build_tensor_noise(df_gesamt)
    df_tasks = build_tasks(df_gesamt)

    df_gesamt.to_csv(config.DATA_DIR / "kombinationen.csv", index=False)
    df_tasks.to_csv(config.DATA_DIR / "tasks.csv", index=False)
    np.save(config.DATA_DIR / "tensor_noise.npy", tensor_noise)

    print(f"Kombinationen: {len(df_gesamt)}")
    print(f"Tasks gesamt (Kombinationen x {config.N_SAMPLE_STAT} Tensorbloecke): {len(df_tasks)}")
    print(f"Tensor-Shape: {tensor_noise.shape}")
    print(f"Geschrieben nach: {config.DATA_DIR}")

    # Schalter kommt standardmaessig aus config.SQS_PUSH_ENABLED (True/False) -
    # per CLI-Flag explizit erzwingbar/abschaltbar, falls mal ein einzelner
    # Lauf abweichend vom aktuellen config.py-Stand gebraucht wird.
    push_enabled = config.SQS_PUSH_ENABLED if sqs_push_enabled is None else sqs_push_enabled
    if push_enabled:
        queue_url = sqs_queue_url or config.SQS_QUEUE_URL
        region = aws_region or config.AWS_REGION
        if not queue_url:
            raise ValueError(
                "SQS_PUSH_ENABLED=True, aber keine Queue-URL gesetzt "
                "(weder config.SQS_QUEUE_URL noch --sqs-queue-url)."
            )
        print("Schreibe Tasks zusaetzlich in SQS-Queue...")
        push_tasks_to_sqs(df_tasks, queue_url, region=region)
    else:
        print("SQS-Push deaktiviert (config.SQS_PUSH_ENABLED=False) - nur lokale Dateien geschrieben.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sqs-queue-url", default=None,
        help="Ueberschreibt config.SQS_QUEUE_URL fuer diesen Lauf.",
    )
    parser.add_argument(
        "--aws-region", default=None,
        help="Ueberschreibt config.AWS_REGION fuer diesen Lauf.",
    )
    parser.add_argument(
        "--push-to-sqs", dest="push_to_sqs", action="store_true", default=None,
        help="Erzwingt SQS-Push fuer diesen Lauf, unabhaengig von config.SQS_PUSH_ENABLED.",
    )
    parser.add_argument(
        "--no-push-to-sqs", dest="push_to_sqs", action="store_false",
        help="Unterdrueckt SQS-Push fuer diesen Lauf, unabhaengig von config.SQS_PUSH_ENABLED.",
    )
    args = parser.parse_args()
    main(sqs_queue_url=args.sqs_queue_url, aws_region=args.aws_region, sqs_push_enabled=args.push_to_sqs)
