"""Aggregierter Fortschritts-Ueberblick ueber mehrere Instanzen hinweg.

Liest die status.json-Dateien, die run_instance.py laufend schreibt.
Kann waehrend die Instanzen noch rechnen wiederholt aufgerufen werden.

    python Main_Prozess/check_progress.py --pattern "data/instances/instance_*/results/status.json"
    (aus der ROBO_Benchmark-Wurzel)

Mit --live zusaetzlich pro Worker-Prozess anzeigen, welche Task (Kombination +
Tensorblock) er gerade bearbeitet und bei welcher Iteration er steht (liest
die worker_<pid>.json-Dateien aus dem workers/-Unterordner neben status.json):

    python Main_Prozess/check_progress.py --pattern "data/instances/instance_*/results/status.json" --live
"""

import argparse
import glob
import json
from pathlib import Path


def print_status_table(files):
    total_done = total_failed = total_all = 0
    print(f"{'Instanz':<20}{'erledigt':>10}{'gesamt':>10}{'%':>8}{'Tasks/s':>10}{'ETA (min)':>12}")
    for f in files:
        with open(f) as fh:
            s = json.load(fh)
        # status.json liegt unter .../instance_X/results/status.json
        name = Path(f).parents[1].name
        eta_min = s["eta_seconds"] / 60 if s.get("eta_seconds") else None
        eta_str = f"{eta_min:.1f}" if eta_min is not None else "-"
        print(f"{name:<20}{s['done']:>10}{s['total']:>10}{s['percent']:>7.1f}%{s['tasks_per_second']:>10.2f}{eta_str:>12}")
        total_done += s["done"]
        total_failed += s["failed"]
        total_all += s["total"]

    pct = 100 * total_done / total_all if total_all else 0
    print("-" * 70)
    print(f"GESAMT: {total_done}/{total_all} ({pct:.1f}%) | Fehlerhafte Tasks: {total_failed}")


def print_live_table(status_files):
    """Pro status.json-Fundstelle die Worker-Live-Dateien im Unterordner
    workers/ einlesen und anzeigen, welche Task/Iteration gerade laeuft."""
    rows = []
    for status_file in status_files:
        instance_name = Path(status_file).parents[1].name
        workers_dir = Path(status_file).parent / "workers"
        for wf in sorted(workers_dir.glob("worker_*.json")):
            try:
                with open(wf) as fh:
                    w = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue  # gerade mitten im atomaren Replace erwischt - naechstes Mal wieder lesbar
            rows.append((instance_name, w))

    if not rows:
        print("Keine laufenden Worker gefunden (workers/-Ordner leer oder noch keine Task gestartet).")
        return

    header = (f"{'Instanz':<14}{'PID':>8}{'Task':>7}{'Komb.':>7}{'Block':>7}  "
              f"{'Surrogat':<28}{'Akqu.':<10}{'Iteration':>12}  {'Status':<8}{'Update'}")
    print(header)
    for instance_name, w in sorted(rows, key=lambda r: (r[0], r[1]["pid"])):
        iter_str = f"{w['iteration']}/{w['max_iteration']}"
        print(f"{instance_name:<14}{w['pid']:>8}{w['task_id']:>7}{w['combination_id']:>7}{w['tensor_block']:>7}  "
              f"{w['surrogate_model']:<28}{w['acquisition_model']:<10}{iter_str:>12}  "
              f"{w['task_status']:<8}{w['last_update']}")


def main(pattern, live):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"Keine status.json gefunden fuer Pattern: {pattern}")
        return

    print_status_table(files)

    if live:
        print()
        print_live_table(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--live", action="store_true",
                         help="zusaetzlich anzeigen, welcher Worker (PID) gerade welche Task/Iteration bearbeitet")
    args = parser.parse_args()
    main(args.pattern, args.live)
