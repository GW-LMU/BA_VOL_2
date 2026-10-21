import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from preprocessing import config
from preprocessing import seed_manager as sdm
from preprocessing.coco_setup import get_problem, evaluate_bbob_function
from postprocessing.metrics import BBOBBenchmark, compute_all_metrics

from main_processing.scenarios.nominal import run_nominal_function, evaluate_nominal_function
from main_processing.scenarios.prior_error import run_prior_error_function, evaluate_prior_error_function
from main_processing.scenarios.stochastic_input import run_stochastic_input_function, evaluate_stochastic_input_function


# Namen der Metriken aus postprocessing.metrics.compute_all_metrics -> Spalten M1 ... M6
METRIC_COLUMNS = {
    "nominal_regret": "M1",
    "expected_performance": "M2",
    "worst_case_performance": "M3",
    "shift_performance": "M4",
    "dro_performance": "M5",
    "cvar": "M6",
}


# Ergebnisspalten pro Iteration (nach den Experimentparametern), feste Reihenfolge fuer die CSV
RESULT_COLUMNS = [
    "iteration", "next_x", "next_y", "next_f_true", "acq_value",
    "best_observed_so_far", "best_true_so_far", "simple_regret",
    "instantaneous_regret", "cumulative_regret", "x_rec",
    *METRIC_COLUMNS.values(),
    # Zusatzinfos, um Modell und Empfehlung nachtraeglich rekonstruieren zu koennen
    "prior_offset", "n_train", "rec_rule", "mu_train", "sigma_train", "hyperparameters",
    "iter_time_s",
    # Startdaten - nur in der Zeile mit iteration == 1 gefuellt
    "start_X", "start_f_obs", "start_f_true",
    "status", "error",
]


def _to_json(value):
    # Listen/Dicts/Arrays als JSON-String in die CSV; auslesen mit json.loads
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return json.dumps(value, default=float)


def select_and_run_variant(surrogate_model, acquisition_func, train_X, train_Y, dim,
                           lower_bound, upper_bound, prior_offset, rng):
    # prior_offset: float (Szenario prior_error) oder None (alle anderen Szenarien)
    # rng: np.random.Generator fuer Verfahren mit eigener Zufallssuche (PROBO)
    # Rueckgabe: (next_x, acq_value, model_info) mit model_info["posterior_mean"]

    if surrogate_model == "GP_BO":
        from main_processing.algorithms import gp_bo
        return gp_bo.run_gaussian_process(train_X, train_Y, dim, acquisition_func,
                                          lower_bound, upper_bound, prior_offset)
    elif surrogate_model == "PROBO":
        from main_processing.algorithms import probo
        return probo.run_imprecise_gausian_prozess(train_X, train_Y, dim, acquisition_func,
                                                   lower_bound, upper_bound,
                                                   prior_offset=prior_offset, rng=rng)
    elif surrogate_model in ("RP_BO", "STABLEOPT", "DRBO", "AIRBO"):
        raise NotImplementedError(f"Surrogatmodell {surrogate_model} ist noch nicht implementiert.")
    else:
        raise ValueError(f"Unbekanntes Surrogatmodell: {surrogate_model}")


def run_and_select_scenario(scenario, intensity):
    # Rueckgabe: (Starttensor, prior_offsets oder None)
    if scenario == "nominal":
        return run_nominal_function(scenario, intensity)
    elif scenario == "prior_error":
        return run_prior_error_function(scenario, intensity)
    elif scenario == "stochastic_input":
        return run_stochastic_input_function(scenario, intensity)
    elif scenario in ("adversarial_input", "distribution_shift"):
        raise NotImplementedError(f"Szenario {scenario} ist noch nicht implementiert.")
    else:
        raise ValueError(f"Unbekanntes Szenario: {scenario}")


# Gibt den ROHEN BBOB-Wert f zurueck (COCO minimiert). Negiert wird nur im Runner.
def select_evaluate_scenario(next_x, scenario, intensity, objective, rng):
    if scenario == "nominal":
        return evaluate_nominal_function(next_x, intensity, objective, rng)
    elif scenario == "prior_error":
        return evaluate_prior_error_function(next_x, intensity, objective, rng)
    elif scenario == "stochastic_input":
        return evaluate_stochastic_input_function(next_x, intensity, objective, rng)
    elif scenario in ("adversarial_input", "distribution_shift"):
        raise NotImplementedError(f"Szenario {scenario} ist noch nicht implementiert.")
    else:
        raise ValueError(f"Unbekanntes Szenario: {scenario}")


def run_single_experiment(experiment):

    surrogate_model = experiment["surrogate_model"]
    acquisition_func = experiment["acquisition_func"]
    function = experiment["function"]
    dimension = experiment["dimension"]
    instance = experiment["instance"]
    scenario = experiment["scenario"]
    intensity = experiment["intensity"]
    repetition = experiment["repetition"]                       # Das ist der Tensor Block
    n_start_samples = experiment["n_start_samples"]
    n_iterations = experiment["budget"] - n_start_samples

    lower_bound, upper_bound = config.BOUNDS

    # Seeds pro Wiederholung, unabhaengig vom Algorithmus (Common Random Numbers):
    #   noise_rng -> Eingaberauschen (nur stochastic_input nutzt ihn)
    #   algo_rng  -> Zufallssuche der Verfahren (PROBO), torch-Seed -> optimize_acqf
    noise_rng = sdm.get_rng("input_noise", repetition)
    algo_rng = sdm.get_rng("algorithm", repetition)
    torch.manual_seed(sdm.ALGORITHM_SEEDS[repetition])

    problem = get_problem(function, dimension, instance)
    try:
        # Zielfunktion dieses Experiments (roher, rauschfreier BBOB-Wert, Minimierung)
        def objective(x):
            return evaluate_bbob_function(x, problem)

        # Metriken laufen immer auf der wahren Funktion
        benchmark = BBOBBenchmark(problem, f_opt=experiment["f_opt"], bounds=config.BOUNDS)

        # Trainingsdaten fuer das Modell laden - Starttensor und ggf. Prior-Fehler aus dem Szenario
        tensor, prior_offsets = run_and_select_scenario(scenario, intensity)

        # Prior-Fehler dieser Wiederholung (bleibt ueber alle Iterationen gleich).
        # Bei nominal / stochastic_input ist prior_offsets None -> kein Prior-Fehler
        prior_offset = None if prior_offsets is None else float(prior_offsets[repetition])

        # train_X: unverrauschte Startpunkte
        # f_obs:   beobachtete f-Werte (bei stochastic_input an x + xi)
        # f_true:  wahre f-Werte an den gespeicherten x (nur fuer die Auswertung)
        train_X_np = np.array(tensor[repetition, :n_start_samples, :dimension], dtype=np.float64)
        f_obs = np.array([
            select_evaluate_scenario(x, scenario, intensity, objective, noise_rng)
            for x in train_X_np
        ])
        f_true = np.array([objective(x) for x in train_X_np])

        # Startdaten fuer die Rekonstruktion (werden nur in Iteration 1 gespeichert)
        start_data = {
            "start_X": _to_json(train_X_np),
            "start_f_obs": _to_json(f_obs),
            "start_f_true": _to_json(f_true),
        }

        rows = []
        cumulative_regret = 0.0
        for iteration in range(1, n_iterations + 1):
            iter_start = time.time()

            # BoTorch maximiert -> Surrogat sieht y = -f
            train_X = torch.as_tensor(train_X_np, dtype=torch.double)
            train_Y = torch.as_tensor(-f_obs, dtype=torch.double).unsqueeze(-1)

            # Naechste Eingabe bestimmen, abhaengig von Surrogatmodell und Akquisitionsfunktion
            next_x, acq_value, model_info = select_and_run_variant(
                surrogate_model, acquisition_func, train_X, train_Y, dimension,
                lower_bound, upper_bound, prior_offset, algo_rng)
            next_x_np = next_x.detach().cpu().numpy().astype(np.float64).reshape(-1)

            # Modell an den Trainingspunkten - nur gespeichert, damit andere
            # Empfehlungsregeln (Posterior-Mittelwert, LCB) nachtraeglich moeglich sind
            posterior_mean = model_info.get("posterior_mean")
            posterior_std = model_info.get("posterior_std")
            n_train = len(train_X_np)
            mu_train = posterior_mean(train_X_np) if posterior_mean is not None else None
            sigma_train = posterior_std(train_X_np) if posterior_std is not None else None

            # Empfohlener Punkt x_rec: Trainingspunkt mit dem besten BEOBACHTETEN Wert
            # (kleinstes f_obs), fuer alle Verfahren gleich (Rodemann & Augustin: best
            # incumbent). next_x ist kein Kandidat, da sein Wert noch nicht beobachtet ist.
            x_rec = train_X_np[int(np.argmin(f_obs))]
            rec_rule = "best_observed"

            # Funktionswert abhaengig vom Szenario (roher f-Wert, ggf. verrauscht)
            next_y = select_evaluate_scenario(next_x_np, scenario, intensity, objective, noise_rng)
            next_f_true = objective(next_x_np)

            # Robustheitsmetriken des empfohlenen Punkts auf der wahren Funktion
            metrics = compute_all_metrics(x_rec, benchmark, config)

            # Neue Daten anhaengen: UNVERRAUSCHTES x, beobachteter Wert
            train_X_np = np.vstack([train_X_np, next_x_np.reshape(1, -1)])
            f_obs = np.append(f_obs, next_y)
            f_true = np.append(f_true, next_f_true)

            # Regret des angefragten Punkts (wahre Funktion) und laufende Summe ueber
            # die BO-Iterationen (Startdesign nicht enthalten, fuer alle Verfahren gleich)
            instantaneous_regret = float(next_f_true - experiment["f_opt"])
            cumulative_regret += instantaneous_regret

            # Experimentparameter + Iterationsergebnisse zusammenfuehren
            rows.append({
                **experiment,
                "iteration": iteration,
                "next_x": next_x_np.tolist(),
                "next_y": float(next_y),                         # beobachtet (Szenario)
                "next_f_true": float(next_f_true),               # rauschfrei
                "acq_value": float(np.asarray(acq_value).reshape(-1)[0]),
                "best_observed_so_far": float(np.min(f_obs)),
                "best_true_so_far": float(np.min(f_true)),
                "simple_regret": float(np.min(f_true) - experiment["f_opt"]),
                "instantaneous_regret": instantaneous_regret,
                "cumulative_regret": cumulative_regret,
                "x_rec": x_rec.tolist(),
                **{METRIC_COLUMNS[name]: value for name, value in metrics.items()},
                # mu_train/sigma_train: Modell (Skala y = -f) an den n_train Trainingspunkten,
                # Reihenfolge = Startpunkte, dann next_x der Iterationen 1 ... iteration-1
                "prior_offset": prior_offset,
                "n_train": n_train,
                "rec_rule": rec_rule,
                "mu_train": _to_json(mu_train),
                "sigma_train": _to_json(sigma_train),
                "hyperparameters": _to_json(model_info.get("hyperparameters")),
                "iter_time_s": time.time() - iter_start,
                **(start_data if iteration == 1 else {}),
                "status": "ok",
                "error": "",
            })
    finally:
        problem.free()

    return rows


def _load_done_task_ids(output_path):
    if output_path is None or not Path(output_path).exists():
        return set()
    done = pd.read_csv(output_path, usecols=["task_id", "status"])
    return set(done.loc[done["status"] == "ok", "task_id"])


def _run_safe(experiment):
    # Laeuft im Worker-Prozess (muss auf Modulebene stehen, damit es pickelbar ist)
    start = time.time()
    try:
        rows = run_single_experiment(experiment)
    except Exception as e:
        # Ein fehlerhaftes Experiment soll nicht den ganzen Lauf abbrechen
        traceback.print_exc()
        rows = [{**experiment, "status": "error", "error": f"{type(e).__name__}: {e}"}]
    return rows, time.time() - start


def _init_worker(threads):
    # Jeder Prozess rechnet mit wenigen Threads, sonst konkurrieren N_JOBS * Kerne Threads
    torch.set_num_threads(threads)


def _resolve_n_jobs(n_jobs, n_experiments):
    # -1 = alle Kerne, -2 = alle bis auf einen, ... (wie bei joblib)
    cpu_count = os.cpu_count() or 1
    if n_jobs is None:
        n_jobs = 1
    elif n_jobs < 0:
        n_jobs = cpu_count + 1 + n_jobs
    return max(1, min(n_jobs, cpu_count, n_experiments))


def _append_to_csv(rows, experiment, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(output_path).exists()
    columns = list(experiment.keys()) + RESULT_COLUMNS
    pd.DataFrame(rows).reindex(columns=columns).to_csv(
        output_path, mode="a", header=write_header, index=False)


def run_experiments(experiments_table, output_path=None, n_jobs=None):
    """Fuehrt alle Experimente aus.

    output_path: optional CSV-Datei. Ergebnisse werden nach JEDEM Experiment
    angehaengt; bereits erfolgreich gerechnete task_ids werden beim Neustart
    uebersprungen (Fortsetzen nach Abbruch).
    n_jobs: Anzahl paralleler Prozesse, Standard config.N_JOBS.
    """

    done = _load_done_task_ids(output_path)
    experiments = [e for e in experiments_table if e["task_id"] not in done]
    if done:
        print(f"{len(done)} Experimente bereits vorhanden, {len(experiments)} verbleiben.")
    if not experiments:
        return pd.DataFrame()

    n_jobs = _resolve_n_jobs(config.N_JOBS if n_jobs is None else n_jobs, len(experiments))
    threads = config.THREADS_PER_JOB
    print(f"Rechne mit {n_jobs} Prozess(en) x {threads} Thread(s) "
          f"({os.cpu_count()} logische Kerne verfuegbar).")

    all_results = []

    def handle_result(i, experiment, rows, duration):
        # Laeuft nur im Hauptprozess -> CSV wird nie von zwei Prozessen gleichzeitig geschrieben
        status = rows[-1]["status"] if rows else "empty"
        print(f"[{i}/{len(experiments)}] {experiment['task_id']}: {status} ({duration:.1f}s)")
        if output_path is not None:
            _append_to_csv(rows, experiment, output_path)
        all_results.extend(rows)

    if n_jobs == 1:
        torch.set_num_threads(threads)
        for i, experiment in enumerate(experiments, start=1):
            rows, duration = _run_safe(experiment)
            handle_result(i, experiment, rows, duration)
    else:
        # BLAS-Threads der Kindprozesse begrenzen (wird beim Import von numpy gelesen)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(threads)

        # Jedes Experiment setzt seine Seeds selbst -> Ergebnisse identisch zum
        # sequentiellen Lauf, nur die Zeilenreihenfolge in der CSV ist anders
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker,
                                 initargs=(threads,)) as pool:
            futures = {pool.submit(_run_safe, e): e for e in experiments}
            for i, future in enumerate(as_completed(futures), start=1):
                experiment = futures[future]
                try:
                    rows, duration = future.result()
                except Exception as e:
                    # z.B. abgestuerzter Worker-Prozess
                    rows = [{**experiment, "status": "error", "error": f"{type(e).__name__}: {e}"}]
                    duration = 0.0
                handle_result(i, experiment, rows, duration)

    # Nach allen Experimenten DataFrame erstellen
    return pd.DataFrame(all_results)
