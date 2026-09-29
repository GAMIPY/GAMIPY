import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from GAMIPY.data.data import Data
from GAMIPY.data.mtypes import (
    ChiSquareMissingnessTest,
    LogisticRegressionMissingnessTest,
    MissingnessTypes,
)


class TestMissingnessTypes(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset with various missingness types
        data = pd.DataFrame({
            'A': [1.0, 2.5, None, 4.0],
            'B': [3.0, 5.5, 6.0, 7.0],
            'C': [8.0, None, None, 10.0]
        })
        self.data = Data(data)

    def test_missingness_types_initialization(self):
        mtypes = MissingnessTypes(self.data)
        self.assertIsInstance(mtypes, MissingnessTypes)

    def test_m_type_raises_without_missing_values(self):
        data = Data(pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': [4.0, 5.0, 6.0]}))
        with self.assertRaises(ValueError):
            MissingnessTypes(data).m_type()

    def test_m_type_multi_column_mcar(self):
        rng = np.random.default_rng(0)
        n = 500
        df = pd.DataFrame({
            'A': rng.normal(size=n),
            'B': rng.normal(size=n),
            'C': rng.normal(size=n),
        })
        df.loc[rng.random(n) < 0.10, 'B'] = np.nan
        df.loc[rng.random(n) < 0.10, 'C'] = np.nan
        self.assertEqual(MissingnessTypes(Data(df)).m_type(), 'MCAR')

    def test_m_type_single_column_mar(self):
        rng = np.random.default_rng(0)
        n = 500
        df = pd.DataFrame({
            'A': rng.normal(size=n),
            'B': rng.normal(size=n),
            'C': rng.normal(size=n),
        })
        df.loc[df['C'] > df['C'].median(), 'B'] = np.nan
        self.assertEqual(MissingnessTypes(Data(df)).m_type(), 'MAR')

    def test_m_type_multi_column_mar(self):
        rng = np.random.default_rng(0)
        n = 500
        df = pd.DataFrame({
            'A': rng.normal(size=n),
            'B': rng.normal(size=n),
            'C': rng.normal(size=n),
            'D': rng.normal(size=n),
        })
        df.loc[df['C'] > df['C'].median(), 'B'] = np.nan
        df.loc[rng.random(n) < 0.10, 'D'] = np.nan
        self.assertEqual(MissingnessTypes(Data(df)).m_type(), 'MAR')

    def test_m_type_single_column_mcar(self):
        rng = np.random.default_rng(0)
        n = 500
        df = pd.DataFrame({
            'A': rng.normal(size=n),
            'B': rng.normal(size=n),
            'C': rng.normal(size=n),
        })
        df.loc[rng.random(n) < 0.15, 'B'] = np.nan
        self.assertEqual(MissingnessTypes(Data(df)).m_type(), 'MCAR')

    def test_m_type_uses_injected_tests(self):
        # A custom test list lets a caller swap in its own strategy without
        # MissingnessTypes knowing about it - the whole point of splitting
        # the tests out into their own Strategy classes.
        class AlwaysMNAR:
            def classify(self, df):
                return 'MNAR'

        self.assertEqual(MissingnessTypes(self.data, tests=[AlwaysMNAR()]).m_type(), 'MNAR')

    def test_m_type_defaults_to_mcar_when_no_test_is_conclusive(self):
        class Inconclusive:
            def classify(self, df):
                return None

        self.assertEqual(MissingnessTypes(self.data, tests=[Inconclusive()]).m_type(), 'MCAR')

    def test_logistic_regression_test_detects_mar(self):
        rng = np.random.default_rng(0)
        n = 500
        df = pd.DataFrame({
            'A': rng.normal(size=n),
            'B': rng.normal(size=n),
            'C': rng.normal(size=n),
        })
        df.loc[df['C'] > df['C'].median(), 'B'] = np.nan
        self.assertEqual(LogisticRegressionMissingnessTest().classify(df), 'MAR')

    def test_logistic_regression_test_defers_on_unpredictable_missingness(self):
        rng = np.random.default_rng(0)
        n = 500
        df = pd.DataFrame({
            'A': rng.normal(size=n),
            'B': rng.normal(size=n),
        })
        df.loc[rng.random(n) < 0.15, 'B'] = np.nan
        self.assertIsNone(LogisticRegressionMissingnessTest().classify(df))

    def test_logistic_regression_test_defers_without_complete_columns(self):
        df = pd.DataFrame({'A': [1.0, None, 3.0], 'B': [None, 2.0, 3.0]})
        self.assertIsNone(LogisticRegressionMissingnessTest().classify(df))

    def test_logistic_regression_test_accepts_injected_classifier(self):
        # No sklearn estimator involved at all - proves the test depends only
        # on the fit/predict_proba interface, not on LogisticRegression itself.
        class StubClassifier:
            def fit(self, X, y):
                self._y = np.asarray(y)
                return self

            def predict_proba(self, X):
                # Memorizes and echoes back the exact labels it was fit on,
                # giving a perfect (AUC=1.0) score regardless of the
                # predictors, so the test should conclude 'MAR'.
                return np.column_stack([1 - self._y, self._y]).astype(float)

        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame({'A': rng.normal(size=n), 'B': rng.normal(size=n)})
        df.loc[rng.random(n) < 0.10, 'B'] = np.nan

        test = LogisticRegressionMissingnessTest(classifier_factory=StubClassifier)
        self.assertEqual(test.classify(df), 'MAR')

    def test_logistic_regression_test_builds_a_fresh_classifier_per_column(self):
        built = []

        def factory():
            built.append(object())
            return LogisticRegression(max_iter=1000)

        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({
            'A': rng.normal(size=n),
            'B': rng.normal(size=n),
            'C': rng.normal(size=n),
        })
        df.loc[rng.random(n) < 0.10, 'B'] = np.nan
        df.loc[rng.random(n) < 0.10, 'C'] = np.nan

        # LogisticRegression used to be un-imported here, so every factory()
        # call raised NameError *after* appending to `built` - swallowed by
        # classify()'s `except Exception`, so this passed without ever
        # fitting a model. Check a real classifier came back too.
        models = []

        def recording_factory():
            model = factory()
            models.append(model)
            return model

        LogisticRegressionMissingnessTest(classifier_factory=recording_factory).classify(df)
        self.assertGreaterEqual(len(built), 1)
        self.assertEqual(len({id(m) for m in models}), len(models))
        self.assertTrue(all(hasattr(m, 'coef_') for m in models))  # actually fitted

    def test_chi_square_test_defers_with_fewer_than_two_missing_columns(self):
        df = pd.DataFrame({'A': [1.0, None, 3.0], 'B': [4.0, 5.0, 6.0]})
        self.assertIsNone(ChiSquareMissingnessTest().classify(df))

    def test_chi_square_test_returns_mcar_or_mnar_with_two_missing_columns(self):
        rng = np.random.default_rng(0)
        n = 300
        df = pd.DataFrame({
            'A': rng.normal(size=n),
            'B': rng.normal(size=n),
            'C': rng.normal(size=n),
        })
        df.loc[rng.random(n) < 0.10, 'B'] = np.nan
        df.loc[rng.random(n) < 0.10, 'C'] = np.nan
        self.assertIn(ChiSquareMissingnessTest().classify(df), ('MCAR', 'MNAR'))
