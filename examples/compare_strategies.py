"""Compare GA operator combinations on the same imputation problem.

Runs several initialization x selection x crossover combinations on the
iris measurements with 10% of the values hidden, and ranks them by mean
NRMSE against the hidden truth (lower is better). All four columns are
numerical, so the numeric-only `arithmetic` and `blend` crossovers can be
included too.

Run from the repository root:  python examples/compare_strategies.py
"""
import random
import time

import numpy as np
import pandas as pd
from evaluate_imputation import hide_values, mean_mode_impute

from GAMIPY import GAMIPY
from GAMIPY.datasets.sample_data import SampleData
from GAMIPY.evaluation import column_errors

COMBINATIONS = [
    # (initialization_strategy, selection_method, crossover_type)
    ("random", "elitist", "uniform"),
    ("random", "tournament", "two_point"),
    ("random", "rank", "arithmetic"),
    ("random", "roulette", "blend"),
    ("random", "steady_state", "one_point"),
    ("hybrid", "tournament", "arithmetic"),
]


def mean_seed(damaged: pd.DataFrame) -> pd.DataFrame:
    """heuristic_population for 'hybrid': every missing cell -> its column mean.

    One row per missing cell: (row label, position among the rows that have a
    missing value, column, value).
    """
    missing_rows = damaged.index[damaged.isna().any(axis=1)]
    row_m = {label: i for i, label in enumerate(missing_rows)}
    return pd.DataFrame([
        (row, row_m[row], col, float(damaged[col].mean()))
        for row in missing_rows
        for col in damaged.columns[damaged.loc[row].isna()]
    ])


def main() -> None:
    original = SampleData().load_data("iris").drop(columns=["class"])
    damaged = hide_values(original, ratio=0.10, seed=1)

    results = []
    for init, selection, crossover in COMBINATIONS:
        random.seed(0)
        np.random.seed(0)
        start = time.perf_counter()
        gamipy = GAMIPY(
            representation='cells',  # these scripts compare the cells GA's operators
            data=damaged,
            population_size=20,
            generations=10,
            initialization_strategy=init,
            heuristic_population=mean_seed(damaged) if init == "hybrid" else None,
            selection_method=selection,
            tournament_size=4,
            crossover_type=crossover,
            crossover_rate=0.8,
            mutation_rate=0.1,
        )
        gamipy.run()
        report = column_errors(original, damaged, gamipy.imputed_data)
        results.append({"initialization": init, "selection": selection, "crossover": crossover,
                        "mean NRMSE": report["NRMSE"].mean(), "mean MAE": report["MAE"].mean(),
                        "seconds": time.perf_counter() - start})

    baseline = column_errors(original, damaged, mean_mode_impute(damaged))
    table = pd.DataFrame(results).sort_values("mean NRMSE").reset_index(drop=True)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(table.to_string(index=False), end="\n\n")
    print(f"mean imputation baseline: mean NRMSE {baseline['NRMSE'].mean():.3f}, "
          f"mean MAE {baseline['MAE'].mean():.3f}")


if __name__ == "__main__":
    main()
