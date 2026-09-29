"""Measure imputation quality on mixed numerical + categorical data.

Hides 10% of the known values in a sample of the bundled `student` dataset,
imputes them with GAMIPY, and compares the result with the hidden truth
using GAMIPY.evaluation.column_errors: RMSE, MAE and NRMSE for numerical
columns, accuracy for categorical ones. A mean/mode imputation is scored the
same way as a baseline.

Run from the repository root:  python examples/evaluate_imputation.py
"""
import random

import numpy as np
import pandas as pd

from GAMIPY import GAMIPY
from GAMIPY.datasets.sample_data import SampleData
from GAMIPY.evaluation import column_errors


def hide_values(df: pd.DataFrame, ratio: float, seed: int) -> pd.DataFrame:
    """Return a copy of `df` with `ratio` of its cells set to NaN (MCAR)."""
    rng = np.random.default_rng(seed)
    return df.mask(rng.random(df.shape) < ratio)


def mean_mode_impute(damaged: pd.DataFrame) -> pd.DataFrame:
    """Baseline: column mean for numerical columns, mode for the others."""
    filled = damaged.copy()
    for col in filled.columns:
        if pd.api.types.is_numeric_dtype(filled[col]):
            filled[col] = filled[col].fillna(filled[col].mean())
        else:
            filled[col] = filled[col].fillna(filled[col].mode().iloc[0])
    return filled


def summary(report: pd.DataFrame) -> str:
    return (f"mean NRMSE {report['NRMSE'].mean():.3f}, "
            f"mean categorical accuracy {report['accuracy'].mean():.1%}")


def main() -> None:
    random.seed(0)
    np.random.seed(0)

    columns = ["school", "sex", "address", "Mjob", "age", "studytime", "absences", "G1", "G2", "G3"]
    original = SampleData().load_data("student")[columns].sample(120, random_state=0).reset_index(drop=True)
    damaged = hide_values(original, ratio=0.10, seed=1)
    print(f"Hid {int(damaged.isna().sum().sum())} of {original.size} values "
          f"({len(columns)} columns, 4 of them categorical)\n")

    gamipy = GAMIPY(
        data=damaged,
        population_size=20,
        generations=10,
        selection_method="tournament",
        tournament_size=4,
        crossover_type="uniform",
        crossover_rate=0.8,
        mutation_rate=0.1,
    )
    gamipy.run()

    # Scored on the hidden cells only (GAMIPY.evaluation).
    ga_report = column_errors(original, damaged, gamipy.imputed_data)
    baseline_report = column_errors(original, damaged, mean_mode_impute(damaged))

    pd.set_option("display.float_format", "{:.3f}".format)
    print("GAMIPY:")
    print(ga_report, end="\n\n")
    print("GAMIPY         :", summary(ga_report))
    print("mean/mode      :", summary(baseline_report))


if __name__ == "__main__":
    main()
