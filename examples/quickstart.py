"""Smallest complete GAMIPY run: impute a DataFrame and look at the result.

Run from the repository root:  python examples/quickstart.py
"""
import random

import numpy as np

from GAMIPY import GAMIPY
from GAMIPY.datasets.sample_data import SampleData


def main() -> None:
    random.seed(0)          # GAMIPY draws from both `random` and NumPy's
    np.random.seed(0)       # global generators; seed both for repeatability

    # Two numerical columns with a missing value, one categorical ("C").
    data = SampleData.mock_data1()
    print("Before imputation:")
    print(data, end="\n\n")

    gamipy = GAMIPY(data=data, population_size=30, generations=15)
    gamipy.run()                       # returns None ...
    imputed = gamipy.imputed_data      # ... the result is stored here

    print("After imputation:")
    print(imputed, end="\n\n")

    # Only the originally missing cells were changed.
    filled = data.isna()
    for row, col in zip(*np.nonzero(filled.to_numpy()), strict=True):
        label, name = data.index[row], data.columns[col]
        print(f"filled {name!r} in row {label}: {imputed.loc[label, name]!r}")


if __name__ == "__main__":
    main()
