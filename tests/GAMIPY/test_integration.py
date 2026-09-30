"""
End-to-end tests that drive GAMIPY(...).run() the way users and
benchmark experiments do, rather than exercising components in isolation
(every earlier integration break - e.g.
heuristic_population never reaching HeuristicPopulation, crossover being
handed a Population instead of a list, the population growing every
generation under steady-state selection - passed the unit tests).
"""
import itertools
import unittest

import numpy as np
import pandas as pd

from GAMIPY.bio.crossover import CROSSOVER_TYPES
from GAMIPY.bio.distance import DISTANCE_METRICS
from GAMIPY.bio.selection import SELECTION_METHODS
from GAMIPY.data.data import Data
from GAMIPY.data.missing_index import MissingIndexLocator
from GAMIPY.datasets.sample_data import SampleData
from GAMIPY.GAMIPY import GAMIPY
from GAMIPY.population.population import POPULATION_STRATEGIES


def mean_seed(df: pd.DataFrame) -> pd.DataFrame:
    """A heuristic_population seeding every missing cell with its column mean."""
    return pd.DataFrame([
        (row, row_m, col, float(df[col].mean()))
        for row, row_m, col in MissingIndexLocator(Data(df)).locate()
    ])


def inject_mcar(df: pd.DataFrame, ratio: float, seed: int) -> pd.DataFrame:
    """Hides each cell independently with probability `ratio` (MCAR)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out = out.mask(rng.random(out.shape) < ratio)
    return out


class IntegrationAssertions(unittest.TestCase):

    def assert_valid_imputation(self, original: pd.DataFrame, gamipy: GAMIPY) -> None:
        imputed = gamipy.imputed_data
        self.assertIsInstance(imputed, pd.DataFrame)
        self.assertEqual(list(imputed.index), list(original.index))
        self.assertEqual(list(imputed.columns), list(original.columns))
        self.assertFalse(imputed.isnull().any().any())
        # Observed cells are never touched.
        observed = original.notna()
        for col in original.columns:
            self.assertEqual(
                list(imputed.loc[observed[col], col]),
                list(original.loc[observed[col], col]),
            )
        self.assertEqual(len(gamipy.population.population), gamipy.config.population_size)


class TestEveryStrategyCombination(IntegrationAssertions):
    """Every initialization_strategy x selection_method x crossover_type runs end to end."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.df = pd.DataFrame({
            'f1': rng.uniform(0, 10, 30),
            'f2': rng.uniform(0, 10, 30),
            'f3': rng.uniform(0, 10, 30),
        }, index=range(100, 130))
        self.df.loc[103, 'f1'] = None
        self.df.loc[107, 'f2'] = None
        self.df.loc[115, 'f3'] = None
        self.df.loc[115, 'f1'] = None

    def test_all_combinations_produce_a_valid_imputation(self):
        combinations = itertools.product(POPULATION_STRATEGIES, SELECTION_METHODS, CROSSOVER_TYPES)
        for init, select, cross in combinations:
            with self.subTest(initialization_strategy=init, selection_method=select, crossover_type=cross):
                original = self.df.copy()
                gamipy = GAMIPY(
                    representation='cells', data=self.df, population_size=12, generations=2,
                    selection_number=4, tournament_size=3,
                    initialization_strategy=init, selection_method=select,
                    crossover_type=cross, heuristic_population=mean_seed(self.df),
                )
                gamipy.run()
                self.assert_valid_imputation(self.df, gamipy)
                pd.testing.assert_frame_equal(self.df, original)  # input not mutated

    def test_every_selectable_fitness_method_runs(self):
        for method in DISTANCE_METRICS:
            if method == 'procrustes':
                continue  # reserved for the covariance comparison
            with self.subTest(fitness_method=method):
                gamipy = GAMIPY(
                    representation='cells', data=self.df, population_size=12, generations=2,
                    selection_number=4, fitness_method=method,
                )
                gamipy.run()
                self.assert_valid_imputation(self.df, gamipy)


class TestMixedNumericalAndCategoricalData(IntegrationAssertions):

    def setUp(self):
        rng = np.random.default_rng(1)
        self.df = pd.DataFrame({
            'num': rng.uniform(0, 10, 30),
            'cat': rng.choice(['red', 'green', 'blue'], 30),
        })
        self.df.loc[4, 'num'] = None
        self.df.loc[9, 'cat'] = None

    def test_label_preserving_crossovers_impute_real_labels(self):
        for cross in ['uniform', 'one_point', 'two_point']:
            for init in ['random', 'zeros', 'ones', 'constant']:
                with self.subTest(crossover_type=cross, initialization_strategy=init):
                    gamipy = GAMIPY(
                        representation='cells', data=self.df, population_size=12, generations=2,
                        selection_number=4, crossover_type=cross,
                        initialization_strategy=init,
                    )
                    gamipy.run()
                    self.assert_valid_imputation(self.df, gamipy)
                    self.assertIn(gamipy.imputed_data.loc[9, 'cat'], ['red', 'green', 'blue'])

    def test_numeric_only_crossovers_reject_categorical_genes(self):
        # Arithmetic/Blend average gene values, which is meaningless for labels.
        for cross in ['arithmetic', 'blend']:
            with self.subTest(crossover_type=cross):
                gamipy = GAMIPY(
                    representation='cells', data=self.df, population_size=12, generations=2,
                    selection_number=4, crossover_type=cross,
                )
                with self.assertRaises(ValueError):
                    gamipy.run()


class TestBundledDatasetPipeline(IntegrationAssertions):
    """An MCAR benchmark run on the bundled iris dataset."""

    def test_iris_with_injected_mcar(self):
        iris = SampleData().load_data('iris')
        features = iris.drop(columns=['class'])
        damaged = inject_mcar(features, ratio=0.02, seed=3)
        self.assertTrue(damaged.isnull().any().any())

        gamipy = GAMIPY(
            representation='cells', data=damaged, population_size=20, generations=3,
            selection_number=5, crossover_rate=0.8, mutation_rate=0.1,
        )
        gamipy.run()
        self.assert_valid_imputation(damaged, gamipy)

        # Random init, uniform crossover and mutation all sample within each
        # column's observed range, so imputed values must stay inside it.
        missing = damaged.isnull()
        for col in features.columns:
            values = gamipy.imputed_data.loc[missing[col], col]
            self.assertTrue((values >= damaged[col].min()).all())
            self.assertTrue((values <= damaged[col].max()).all())

        rmse = np.sqrt(((gamipy.imputed_data[missing] - features[missing]) ** 2).stack().mean())
        self.assertTrue(np.isfinite(rmse))


class TestColumnsWithNoObservedValueInMissingRows(IntegrationAssertions):
    """Initialization draws from m_data (the rows with any missing value).
    When a column has no observed value there - e.g. a single missing cell -
    it used to crash (numeric: OverflowError from uniform(nan, nan);
    categorical: IndexError from random.choice([])) or silently impute None
    (zeros/ones/constant on a categorical column). It now falls back to the
    full data for that column only."""

    def setUp(self):
        rng = np.random.default_rng(2)
        self.df = pd.DataFrame({
            'num': rng.uniform(0, 10, 30),
            'cat': rng.choice(['red', 'green', 'blue'], 30),
        })

    def run_gamipy(self, df, init):
        gamipy = GAMIPY(representation='cells', data=df, population_size=10, generations=2,
                        selection_number=4, initialization_strategy=init)
        gamipy.run()
        self.assert_valid_imputation(df, gamipy)
        return gamipy

    def test_single_missing_numeric_cell(self):
        df = self.df.copy()
        df.loc[5, 'num'] = None
        for init in ['random', 'zeros', 'ones', 'constant']:
            with self.subTest(initialization_strategy=init):
                self.run_gamipy(df, init)

    def test_single_missing_categorical_cell(self):
        df = self.df.copy()
        df.loc[5, 'cat'] = None
        for init in ['random', 'zeros', 'ones', 'constant']:
            with self.subTest(initialization_strategy=init):
                gamipy = self.run_gamipy(df, init)
                self.assertIn(gamipy.imputed_data.loc[5, 'cat'], ['red', 'green', 'blue'])

    def test_column_missing_in_every_row_that_has_a_missing_value(self):
        df = self.df.copy()
        df.loc[[3, 8, 12], 'num'] = None
        self.run_gamipy(df, 'random')

    def test_column_with_no_observed_value_at_all_raises_value_error(self):
        df = self.df.copy()
        df['empty'] = np.nan
        gamipy = GAMIPY(representation='cells', data=df, population_size=10, generations=2, selection_number=4)
        with self.assertRaises(ValueError):
            gamipy.run()

    def test_sampling_still_uses_m_data_when_it_has_observed_values(self):
        # The fallback must not change existing behavior: with observed
        # values in m_data, random init stays inside m_data's own range.
        df = pd.DataFrame({'a': np.arange(30, dtype=float), 'b': np.arange(30, dtype=float)})
        df.loc[[0, 1], 'a'] = None      # m_data rows 0 and 1 ...
        df.loc[[0, 1], 'b'] = [0.5, 1.5]  # ... have b in [0.5, 1.5] only
        df.loc[2, 'b'] = None
        df.loc[2, 'a'] = 2.0
        gamipy = GAMIPY(representation='cells', data=df, population_size=20, initialization_strategy='random',
                        validation_fraction=0.0)
        gamipy.initialize_population()
        for individual in gamipy.population.population:
            for gene in individual.genes:
                if gene.name == 'b':
                    self.assertTrue(0.5 <= gene.data <= 1.5)


if __name__ == '__main__':
    unittest.main()
