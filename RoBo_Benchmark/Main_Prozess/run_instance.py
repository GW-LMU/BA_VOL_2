

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pfade  # noqa: F401 - fuegt Pre_Processing/ (config) und BO_Verfahren/ zu sys.path hinzu

import argparse
import csv
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import cocoex

import config
import core_bo

RESULT_COLUMNS = [
    "task_id", "combination_id", "tensor_block", "funktion", "dimension",
    "surrogate_model", "acquisition_model", "sample_size_initial",
    "max_iteration", "noise_config", "f_opt_approx", "iteration",
    # verrauschte Kennzahlen (wie bisher)
    "best_so_far", "simple_regret", "immediate_regret", "x_distance",
    # rauschfreie Kennzahlen, siehe core_bo.compute_all_metrics() - Regrets
    # daraus berechnet Post_Processing/derived_metrics.py
    "f_true_xt", "best_true_so_far", "f_true_rec",
    "robust_worstcase", "robust_expected", "robust_std",
    "stableopt_worstcase_xtilde", "context_expected", "context_worstcase",
    "status", "error",
]


def _check_header(path):
    """Bricht ab, wenn eine bestehende Ergebnisdatei andere Spalten hat (z.B.
    aus einem Lauf vor Einfuehrung der neuen Kennzahlen) - Anhaengen wuerde
    die Spalten sonst stillschweigend verschieben."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, newline="") as f:
        header = next(csv.reader(f), [])
    if header != RESULT_COLUMNS:
        raise SystemExit(
            f"{path} hat ein anderes Spaltenformat als die aktuelle Pipeline "
            f"(alter Lauf?). Datei verschieben/archivieren oder --output-dir "
            f"auf einen neuen Ordner setzen."
        )

_WORKER_STATE = {}


def _init_worker(tensor_path, bounds, workers_dir):
    # Eigenes Arbeitsverzeichnis pro Prozess: cocoex schreibt bei der
    # Optimum-Abfrage eine Datei mit festem Namen ins cwd - ohne eigenes
    # cwd wuerden sich parallele Worker gegenseitig ueberschreiben.
    worker_dir = Path(tempfile.mkdtemp(prefix=f"bo_worker_{os.getpid()}_"))
    os.chdir(worker_dir)

    _WORKER_STATE["suite"] = cocoex.Suite("bbob", "", "")
    _WORKER_STATE["tensor_noise"] = np.load(tensor_path)
    _WORKER_STATE["bounds"] = bounds
    # eigene Live-Statusdatei je Worker-Prozess, ueber den PID eindeutig,
    # damit sich parallele Worker beim Schreiben nicht in die Quere kommen
    _WORKER_STATE["worker_status_path"] = Path(workers_dir) / f"worker_{os.getpid()}.json"


def _write_worker_status(task_row, iteration, task_status):
    """Schreibt den Live-Fortschritt dieses Worker-Prozesses atomar (tmp + replace,
    analog write_status()), damit check_progress.py nie eine halb geschriebene
    Datei liest.

    Rein informativ (nur fuer check_progress.py) - ein Fehler hier (z.B.
    WinError 5 "Zugriff verweigert" durch Virenscanner/Indexer, der die Datei
    kurz offen haelt) darf NIE einen sonst erfolgreichen Task als "error"
    erscheinen lassen und damit bereits berechnete Ergebniszeilen verwerfen -
    siehe _run_one_task(), wo das Ergebnis von run_task() sonst mit im selben
    try-Block gestanden haette.
    """
    path = _WORKER_STATE.get("worker_status_path")
    if path is None:
        return
    try:
        status = {
            "pid": os.getpid(),
            "task_id": int(task_row["task_id"]),
            "combination_id": int(task_row["combination_id"]),
            "tensor_block": int(task_row["tensor_block"]),
            "funktion": int(task_row["funktion"]),
            "dimension": int(task_row["dimension"]),
            "surrogate_model": task_row["surrogate_model"],
            "acquisition_model": task_row["acquisition_model"],
            "noise_config": task_row.get("noise_config"),
            "iteration": iteration,
            "max_iteration": int(task_row["max_iteration"]),
            "task_status": task_status,  # "laeuft" | "fertig" | "fehler"
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(status, f, indent=2)
        tmp_path.replace(path)
    except OSError as exc:
        # nur eine Konsolenmeldung - der eigentliche Task-Erfolg/-Ergebnis
        # haengt nicht von diesem Live-Statusfile ab
        print(f"[warnung] worker status write fehlgeschlagen (ignoriert): {exc}")


def _run_one_task(task_row):
    suite = _WORKER_STATE["suite"]
    tensor_noise = _WORKER_STATE["tensor_noise"]
    bounds = _WORKER_STATE["bounds"]

    _write_worker_status(task_row, iteration=0, task_status="laeuft")

    def progress_callback(iteration):
        _write_worker_status(task_row, iteration, task_status="laeuft")

    try:
        rows = core_bo.run_task(task_row, suite, tensor_noise, bounds, progress_callback=progress_callback)
        _write_worker_status(task_row, iteration=int(task_row["max_iteration"]), task_status="fertig")
        return rows
    except Exception as exc:  # ein fehlerhafter Task darf den Lauf nicht stoppen
        _write_worker_status(task_row, iteration=-1, task_status="fehler")
        return [{
            **task_row,
            "iteration": -1,
            "best_so_far": None,
            "simple_regret": None,
            "immediate_regret": None,
            "x_distance": None,
            "status": "error",
            "error": str(exc),
        }]


def load_done_task_ids(results_path):
    if not results_path.exists() or results_path.stat().st_size == 0:
        # leere Datei kann von einem abgebrochenen vorherigen Lauf uebrig
        # geblieben sein (z.B. Prozess vor dem Schreiben des Headers
        # abgestuerzt) - dann ist "keine Ergebnisse" die richtige Annahme.
        return set()
    df = pd.read_csv(results_path, usecols=["task_id", "status"])
    return set(df.loc[df["status"] == "ok", "task_id"].unique())


def _dim_list_label():
    """Alle aktiven Dimensionen aus config.DIMENSIONEN als Label fuers
    Dateinamen-Format DIM[...], z.B. "DIM[2,3,5,10,20,40]"."""
    return "DIM[" + ",".join(str(d) for d in config.DIMENSIONEN) + "]"


class PerVerfahrenWriter:
    """Verwaltet die zusaetzlichen Pro-Verfahren-Ergebnisdateien (siehe
    config.RESULT_PER_VERFAHREN_ENABLED/RESULT_PER_VERFAHREN_NAMEN).

    Legt pro (surrogate_model, acquisition_model) verzoegert eine eigene
    Datei unter <output_dir>/<surrogate_model>/ an und haengt Zeilen an -
    zusaetzlich zur zentralen results.csv, die davon unberuehrt bleibt.
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._handles = {}  # (surrogate_model, acquisition_model) -> (file, DictWriter)

    def _writer_for(self, surrogate_model, acquisition_model):
        key = (surrogate_model, acquisition_model)
        if key not in self._handles:
            verfahren_dir = self.output_dir / surrogate_model
            verfahren_dir.mkdir(parents=True, exist_ok=True)
            filename = (
                f"{surrogate_model}_{acquisition_model}_{_dim_list_label()}_"
                f"{config.noise_config_label()}_{self.run_timestamp}.csv"
            )
            file_path = verfahren_dir / filename
            file_exists = file_path.exists()
            f = open(file_path, "a", newline="")
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            if not file_exists:
                writer.writeheader()
            self._handles[key] = (f, writer)
        return self._handles[key]

    def write_row(self, row):
        surrogate_model = row.get("surrogate_model")
        if surrogate_model not in config.RESULT_PER_VERFAHREN_NAMEN:
            return
        f, writer = self._writer_for(surrogate_model, row.get("acquisition_model"))
        writer.writerow({col: row.get(col) for col in RESULT_COLUMNS})
        f.flush()

    def close(self):
        for f, _ in self._handles.values():
            f.close()


def write_status(status_path, done, total, failed, start_time):
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = total - done
    eta_seconds = remaining / rate if rate > 0 else None
    status = {
        "done": done,
        "failed": failed,
        "total": total,
        "percent": round(100 * done / total, 2) if total else 0.0,
        "elapsed_seconds": round(elapsed, 1),
        "tasks_per_second": round(rate, 3),
        "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
        "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp_path = status_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(status, f, indent=2)
    tmp_path.replace(status_path)


def main(instance_dir, workers, output_dir):
    instance_dir = Path(instance_dir)
    tasks_path = instance_dir / "tasks.csv"
    # absolut, da die Worker-Prozesse ihr cwd wechseln (siehe _init_worker)
    tensor_path = (instance_dir / "tensor_noise.npy").resolve()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"
    status_path = output_dir / "status.json"
    # Live-Fortschritt je Worker-Prozess (welcher Task, welche Iteration) -
    # eigener Unterordner, damit check_progress.py ihn separat vom
    # aggregierten status.json einlesen kann
    workers_dir = output_dir / "workers"
    workers_dir.mkdir(parents=True, exist_ok=True)
    for stale in workers_dir.glob("worker_*.json"):
        stale.unlink()  # Leichen von einem frueheren Lauf (andere PIDs) entfernen

    _check_header(results_path)
    df_tasks = pd.read_csv(tasks_path)
    done_ids = load_done_task_ids(results_path)
    pending = df_tasks[~df_tasks["task_id"].isin(done_ids)]

    total = len(df_tasks)
    print(f"Tasks gesamt: {total} | bereits erledigt: {len(done_ids)} | offen: {len(pending)}")

    if len(pending) == 0:
        write_status(status_path, len(done_ids), total, 0, time.time())
        print("Nichts zu tun - alle Tasks bereits erledigt.")
        return

    bounds = (config.MINUS_AREA, config.PLUS_AREA)

    file_exists = results_path.exists()
    results_file = open(results_path, "a", newline="")
    writer = csv.DictWriter(results_file, fieldnames=RESULT_COLUMNS)
    if not file_exists:
        writer.writeheader()

    # Zusaetzliche Pro-Verfahren-Ablage (siehe config.RESULT_PER_VERFAHREN_ENABLED) -
    # schreibt NUR zusaetzlich, die zentrale results.csv oben bleibt unveraendert.
    per_verfahren_writer = PerVerfahrenWriter(output_dir) if config.RESULT_PER_VERFAHREN_ENABLED else None

    start_time = time.time()
    done = len(done_ids)
    failed = 0

    task_dicts = pending.to_dict("records")

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(tensor_path, bounds, workers_dir.resolve()),
    ) as pool:
        futures = [pool.submit(_run_one_task, t) for t in task_dicts]

        with tqdm(total=len(futures), desc=f"instance {instance_dir.name}") as pbar:
            for future in as_completed(futures):
                rows = future.result()
                for row in rows:
                    writer.writerow({col: row.get(col) for col in RESULT_COLUMNS})
                    if per_verfahren_writer is not None:
                        per_verfahren_writer.write_row(row)
                results_file.flush()

                if rows and rows[0].get("status") == "error":
                    failed += 1
                done += 1
                pbar.update(1)

                if done % 5 == 0 or done == total:
                    write_status(status_path, done, total, failed, start_time)

    write_status(status_path, done, total, failed, start_time)
    results_file.close()
    if per_verfahren_writer is not None:
        per_verfahren_writer.close()
    print(f"Fertig. Ergebnisse in: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-dir", required=True, help="Ordner mit tasks.csv + tensor_noise.npy")
    parser.add_argument("--workers", type=int, default=config.WORKERS_PRO_INSTANZ)
    parser.add_argument("--output-dir", default=None, help="Standard: <instance-dir>/results")
    args = parser.parse_args()

    output_dir = args.output_dir or (Path(args.instance_dir) / "results")
    main(args.instance_dir, args.workers, output_dir)
