"""Berechnet die Regrets aus den rauschfreien Kennzahlen in results.csv.

run_task() speichert bewusst nur WERTE (f_true_rec, best_true_so_far,
f_true_xt, ...), keine Regrets - so koennen Wert und Regret nie durch einen Bug
auseinanderlaufen. Die Regrets entstehen erst hier aus f_opt_approx (exakt aus
COCO, siehe prepare_tasks.get_bbob_fopt_exact):

    regret_true            = f_opt - f_true_rec        Hauptkennzahl: Regret der Empfehlung
    regret_true_observed   = f_opt - best_true_so_far  bester je getroffener Punkt
    immediate_regret_true  = f_opt - f_true_xt         Guete des aktuellen Vorschlags

Aeltere results.csv ohne diese Spalten bleiben unveraendert.
"""

DERIVED = {
    "regret_true": "f_true_rec",
    "regret_true_observed": "best_true_so_far",
    "immediate_regret_true": "f_true_xt",
}


def add_derived_metrics(df):
    if "f_opt_approx" not in df.columns:
        return df
    for target, source in DERIVED.items():
        if source in df.columns:
            df[target] = df["f_opt_approx"] - df[source]
    return df
