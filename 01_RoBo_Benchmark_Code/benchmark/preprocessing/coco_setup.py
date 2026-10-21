import glob
import re
import shutil
import uuid
from functools import lru_cache

import cocoex
import numpy as np

cocoex.log_level("warning")      # keine "COCO INFO"-Zeilen pro f_opt-Abfrage

# Holenbzw. bereitstellen der Testumgebung (BBOB-Suite) 
@lru_cache(maxsize=1)
def get_suite():
    # Eine BBOB-Suite fuer den ganzen Prozess (ohne Observer -> keine COCO-Logs)
    return cocoex.Suite("bbob", "", "")

# Basiend auf den Suite holt wir sich die Benachmarkfunktion geholt 
def get_problem(funktion, dimension, instance=1, suite=None):
    suite = get_suite() if suite is None else suite
    return suite.get_problem_by_function_dimension_instance(
        function=funktion, dimension=dimension, instance=instance
    )

# Holt sich das fopt der Benchmarkfunktion (BBOB) und gibt es zurück.
# Idee von Claude, sich das fopt einmalig über dat-Datei zu holen, da es nicht direkt von cocoex bereitgestellt wird.
@lru_cache(maxsize=None)
def get_bbob_fopt_exact(funktion, dimension, instance=1):
    suite = cocoex.Suite("bbob", "", "")
    problem = suite.get_problem_by_function_dimension_instance(
        function=funktion, dimension=dimension, instance=instance)
    
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
    with open(dat_files[0]) as fh:
        first_line = fh.readline()
    match = re.search(r"Fopt \(([-\d.eE+]+)\)", first_line)
    if match is None:
        raise RuntimeError(
            f"Konnte 'Fopt (...)' nicht aus der .dat-Kopfzeile lesen: {first_line!r}"
        )
    fopt_raw = float(match.group(1))

    shutil.rmtree(f"exdata/{result_folder}", ignore_errors=True)

    return fopt_raw                 ### WIchtig hier : Dieser Wert ist ein Minimum und kein Maximun  => muss nich gegiert werden 


def evaluate_bbob_function(x, problem):
    # Roher BBOB-Wert f(x) (Minimierung) fuer EINEN Punkt
    x_numpy = np.asarray(x, dtype=float).reshape(-1)
    return float(problem(x_numpy))
