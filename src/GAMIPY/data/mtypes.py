from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from GAMIPY.data.data import Data


class MissingnessTest(ABC):

    """
    A single missingness-type test strategy.

    Each concrete test looks at one statistical signal (e.g. whether
    missingness is predictable from observed columns, or whether the
    missingness pattern across columns has structure) and either reaches a
    conclusion or defers. `MissingnessTypes.m_type()` runs a list of these
    in order and takes the first non-None result, so a test never needs to
    know about the others, and a new test can be added without touching the
    existing ones (each test is one Single-Responsibility, Open/Closed
    strategy class rather than a branch in a shared method).
    """

    @abstractmethod
    def classify(self, df: pd.DataFrame) -> "str | None":

        """
        Attempts to classify the missingness type of `df`.

        Args
        ----
        - **df (pd.DataFrame)**: The dataset to test. Assumed to already
          contain at least one missing value (`MissingnessTypes.m_type()`
          guarantees this before calling any test).

        Returns
        -------
        - **str | None**: 'MCAR', 'MAR', or 'MNAR' if this test reaches a
          conclusion on this data; `None` if it cannot run or is
          inconclusive, so the caller should defer to the next test.
        """

        pass


class LogisticRegressionMissingnessTest(MissingnessTest):

    """
    MAR test: is any column's missingness predictable from the fully-observed
    columns?

    For each column that has missing values, a logistic regression tries to
    predict that column's missingness indicator from the columns that are
    never missing. Model quality is judged by ROC-AUC rather than raw
    accuracy, since accuracy is misleading whenever the missing rate is low
    (a model that always predicts "not missing" already scores high on
    accuracy without being predictive of anything). If any column's
    missingness is predictable above `auc_threshold`, the dataset is
    classified 'MAR'; otherwise this test defers (returns `None`) rather
    than guessing MCAR vs. MNAR, which it has no signal for.

    The underlying classifier isn't hardcoded to `LogisticRegression`: this
    test depends only on the `fit(X, y)` / `predict_proba(X)` interface, and
    `classifier_factory` is called fresh for every column so no state (e.g.
    a previous column's fitted coefficients) leaks between columns. A
    caller can inject any classifier honouring that interface - e.g. a
    stub in tests, or a different sklearn estimator - without editing this
    class.

    Attributes
    ----------
    - **auc_threshold (float)**: Minimum ROC-AUC for a column's missingness
      to be considered predictable from the observed data (default: 0.75).
    - **classifier_factory (Callable[[], object])**: Builds a fresh,
      unfitted classifier for each column tested. Defaults to
      `lambda: LogisticRegression(max_iter=1000)`.
    """

    def __init__(
        self,
        auc_threshold: float = 0.75,
        classifier_factory: "Callable[[], Any]" = lambda: LogisticRegression(max_iter=1000),
    ) -> None:
        self.auc_threshold = auc_threshold
        self.classifier_factory = classifier_factory

    def classify(self, df: pd.DataFrame) -> "str | None":

        complete_cols = df.columns[~df.isna().any()]
        incomplete_cols = df.columns[df.isna().any()]
        if len(complete_cols) == 0:
            return None

        for col in incomplete_cols:
            target = df[col].isna().astype(int)
            predictors = df[complete_cols].dropna()
            target = target.loc[predictors.index]
            if target.nunique() < 2:
                continue
            try:
                model = self.classifier_factory()
                model.fit(predictors, target)
                proba = model.predict_proba(predictors)[:, 1]
                if roc_auc_score(target, proba) > self.auc_threshold:
                    return 'MAR'
            except Exception:
                continue

        return None


class ChiSquareMissingnessTest(MissingnessTest):

    """
    MCAR/MNAR test: does the missingness pattern across columns have
    structure?

    With two or more columns that have missing values, a chi-square test of
    independence checks whether their missingness indicators are
    independent of each other: a low p-value rejects independence (there IS
    structure -> 'MNAR'), a high p-value is consistent with 'MCAR'. With
    fewer than two missing columns, or too little data for the test, this
    defers (returns `None`).

    Attributes
    ----------
    - **alpha (float)**: Significance level for the independence test
      (default: 0.05).
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def classify(self, df: pd.DataFrame) -> "str | None":

        incomplete_cols = df.columns[df.isna().any()]
        if len(incomplete_cols) < 2:
            return None

        missing_matrix = df[incomplete_cols].isna().astype(int)
        filtered = missing_matrix.loc[(missing_matrix.sum(axis=1) > 0), :]
        if filtered.shape[0] < 2:
            return None

        _, p, _, _ = chi2_contingency(filtered)
        return 'MCAR' if p >= self.alpha else 'MNAR'


class MissingnessTypes:

    """
    Determines the type of missingness (MCAR, MAR, or MNAR) in a dataset by
    running a list of MissingnessTest strategies in order and returning the
    first conclusive result.

    Attributes
    ----------
    - **data (Data)**: The dataset to be analyzed for missingness types.
    - **tests (list[MissingnessTest])**: The strategies to try, in order.
      Defaults to `[LogisticRegressionMissingnessTest(), ChiSquareMissingnessTest()]`.
      A caller can inject its own list - e.g. to tune thresholds or swap in
      a different test - without MissingnessTypes needing to know about any
      concrete test implementation.
    """

    def __init__(self, data: Data, tests: "list[MissingnessTest] | None" = None):

        """
        Initializes the MissingnessTypes object with a dataset.

        Args
        ----
        - **data (Data)**: The dataset containing the data to be analyzed for missingness types.
        - **tests (list[MissingnessTest] | None)**: The ordered strategies to try. Defaults to the
          logistic-regression MAR test followed by the chi-square MCAR/MNAR test.

        Raises
        ------
        - **ValueError**: If the input data is not a valid Data object.
        """

        if not isinstance(data, Data):
            raise ValueError("Input data must be a Data object.")
        self.data = data
        self.tests = tests if tests is not None else [
            LogisticRegressionMissingnessTest(),
            ChiSquareMissingnessTest(),
        ]

    def m_type(self) -> str:

        """
        Determines the type of missingness in the dataset by running `self.tests`
        in order and returning the first conclusive result.

        If none of the tests reach a conclusion (e.g. a single missing
        column whose missingness isn't predictable from observed data), the
        result defaults to 'MCAR'. This is an explicit assumption, not a
        proven result: MNAR, by definition, depends on the missing value
        itself, which can never be observed, so MCAR vs. MAR vs. MNAR can
        never be proven purely from the observed data (Rubin, 1976).

        Returns
        -------
        - **str**: One of 'MCAR', 'MAR', or 'MNAR'.

        Raises
        ------
        - **ValueError**: If the dataset has no missing values.
        """

        # A diagnostic of the data; the GA itself doesn't use it.
        df = self.data.data
        if not df.isna().any().any():
            raise ValueError("No missing values found; missingness type is not applicable.")

        for test in self.tests:
            result = test.classify(df)
            if result is not None:
                return result

        return 'MCAR'
