
# preprocessing/experiment_design.py

from itertools import product
from preprocessing import config
from preprocessing.coco_setup import get_bbob_fopt_exact

# Hier werden die einzele Szenarios Intensitäten geladen mithilfe später Senitifitätsanalysen geabut werden können 
# Alles bis auf adversarial_input und distribution_shift aktiviert 
def get_scenario_intensities(scenario):

    scenario_intensities = {
        "nominal": [0.0],
        "prior_error": config.PRIOR_SIGMAS,
        "stochastic_input": config.INPUT_SIGMAS,
        #"adversarial_input": config.ADVERSARIAL_EPSILONS,
        #"distribution_shift": config.SHIFT_MAGNITUDES
    }
    return scenario_intensities[scenario]





def create_experiment_table():


    experiments = []

    for scenario in config.SCENARIOS:

        intensities = get_scenario_intensities(scenario)

        for (
            algorithm,
            function,
            dimension,
            instance,
            intensity,
            repetition
        ) in product(
            config.ALGORITHMS,
            config.FUNCTIONS,
            config.DIMENSIONS,
            config.INSTANCES,
            intensities,
            range(config.N_WIEDERHOLUNG)
        ):

            task_id = (
                f"{algorithm}"
                f"_f{function}"
                f"_d{dimension}"
                f"_i{instance}"
                f"_{scenario}"
                f"_level{intensity:g}"
                f"_r{repetition:02d}"
            )

            experiment = {
                "task_id": task_id,
                "algorithm": algorithm,
                "surrogate_model": config.ALGORITHM_CONFIGS[algorithm]["surrogate_model"],
                "acquisition_func": config.ALGORITHM_CONFIGS[algorithm]["acquisition"],
                "function": function,
                "dimension": dimension,
                "instance": instance,
                "scenario": scenario,
                "intensity": float(intensity),
                "repetition": repetition,
                "n_start_samples": config.N_START_SAMPLES,
                "budget": config.EVALUATION_BUDGET,
                "f_opt": get_bbob_fopt_exact(function, dimension, instance),   # roh, Minimierung
            }

            experiments.append(experiment)

    return experiments





# Intensität 0.0 bei einem Störungsszenario entspricht dem nominalen Fall.
    # Diese Läufe werden entfernt, damit der nominale Fall nur einmal gerechnet wird.
def remove_duplicate_nominal_runs(experiments):
    filtered_experiments = [
        experiment
        for experiment in experiments
        if experiment["scenario"] == "nominal" or experiment["intensity"] != 0.0]
    return filtered_experiments
