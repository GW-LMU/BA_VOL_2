# Startpunkt des Benchmarks: erstellt den Experimentplan und fuehrt die Experimente durch.
# Aufruf aus dem Ordner benchmark/:
#     python main.py                      # kompletter Plan
#     python main.py --max-experiments 3  # kurzer Testlauf
#     python main.py --jobs 4             # Anzahl Prozesse (ueberschreibt config.N_JOBS)

import argparse
import os
from pathlib import Path

from preprocessing import config

# BLAS-Threads begrenzen, BEVOR numpy/torch importiert werden. Gilt auch fuer die
# Worker-Prozesse -> sequentieller und paralleler Lauf rechnen bitgleich.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = str(config.THREADS_PER_JOB)
from preprocessing.experiment_design import create_experiment_table, remove_duplicate_nominal_runs
from main_processing.experiment_runner import run_experiments
from postprocessing.excel_export import export_excel


def main(max_experiments=None, output_path=None, n_jobs=None):
    # 1. Experimentplan erstellen (Intensitaet 0.0 der Stoerszenarien = nominal -> entfernen)
    experiments = create_experiment_table()
    experiments = remove_duplicate_nominal_runs(experiments)
    if max_experiments is not None:
        experiments = experiments[:max_experiments]

    print(f"{len(experiments)} Experimente geplant.")

    # 2. Experimente durchfuehren (Ergebnisse werden laufend in output_path geschrieben)
    output_path = output_path or config.RESULTS_DIR / "raw" / "results.csv"
    results_df = run_experiments(experiments, output_path=output_path, n_jobs=n_jobs)

    print(f"Fertig: {len(results_df)} Zeilen, gespeichert in {output_path}")

    # 3. Excel aus der kompletten CSV erstellen (inkl. frueherer, fortgesetzter Laeufe)
    if Path(output_path).exists():
        xlsx_path = export_excel(output_path)
        print(f"Excel erstellt: {xlsx_path}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args()
    main(args.max_experiments, args.output, args.jobs)
