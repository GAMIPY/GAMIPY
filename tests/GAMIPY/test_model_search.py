"""representation='model' (GAMIPY.model_search): the GA evolves per-column
linear prediction models instead of cell values."""
import random
import unittest

import numpy as np
import pandas as pd

from GAMIPY import GAMIPY
from GAMIPY.config import GAConfig
from GAMIPY.data.data import Data
from GAMIPY.model_search import ModelSearch


def mixed_frame(seed=0, rows=120):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=rows)
    df = pd.DataFrame({
        'a': a,
        'b': 3 * a + rng.normal(scale=0.2, size=rows),
        'c': rng.uniform(0, 5, rows),
        'k': np.where(a > 0, 'hi', 'lo'),
    })
    return df, df.mask(rng.random(df.shape) < 0.15)


def seeded():
    random.seed(0)
    np.random.seed(0)


def search(df, **options):
    settings = {'population_size': 30, 'generations': 40, 'tournament_size': 4, 'crossover_rate': 0.8,
                'mutation_rate': 0.1, **options}
    config = GAConfig(**settings)
    return ModelSearch(config, Data(df))


class TestModelSearch(unittest.TestCase):

    def test_fills_every_cell_and_keeps_observed_ones(self):
        _, damaged = mixed_frame()
        seeded()
        result = search(damaged).run()
        self.assertFalse(result.isna().any().any())
        observed = damaged.notna()
        pd.testing.assert_frame_equal(result[observed], damaged[observed])
        self.assertTrue(set(result['k']) <= {'hi', 'lo'})

    def test_learns_a_strong_linear_relation(self):
        # b = 3a + noise: a model beats the column mean by far on b.
        original, damaged = mixed_frame(seed=1)
        seeded()
        result = search(damaged, generations=80).run()
        hidden = damaged['b'].isna() & original['b'].notna()
        model_error = ((result.loc[hidden, 'b'] - original.loc[hidden, 'b']) ** 2).mean()
        mean_error = ((damaged['b'].mean() - original.loc[hidden, 'b']) ** 2).mean()
        self.assertLess(model_error, 0.5 * mean_error)

    def test_a_model_never_uses_its_own_column(self):
        _, damaged = mixed_frame()
        model = search(damaged)
        for col in model.targets:
            for out in model.output_slots[col]:
                self.assertTrue((model.allowed[out, model.feature_slots[col]] == 0).all())

    def test_same_numpy_seed_same_result(self):
        _, damaged = mixed_frame()
        seeded()
        first = search(damaged).run()
        seeded()
        second = search(damaged).run()
        pd.testing.assert_frame_equal(first, second)

    def test_patience_stops_early(self):
        _, damaged = mixed_frame()
        seeded()
        model = search(damaged, generations=200, patience=3)
        model.run()
        self.assertLess(model.generations_run, 200)


class TestModelThroughGAMIPY(unittest.TestCase):

    def test_run(self):
        _, damaged = mixed_frame()
        seeded()
        gamipy = GAMIPY(data=damaged, population_size=20, generations=10, representation='model')
        gamipy.run()
        self.assertFalse(gamipy.imputed_data.isna().any().any())
        self.assertIsNotNone(gamipy.model)

    def test_model_is_the_default(self):
        self.assertEqual(GAConfig().representation, 'model')
        self.assertIsNotNone(GAMIPY(data=mixed_frame()[1]).model)

    def test_a_column_with_one_observed_value_is_filled(self):
        df = pd.DataFrame({'a': [1.0, 2, 3, 4, 5, 6], 'b': [np.nan, 5, np.nan, np.nan, np.nan, np.nan],
                           'c': ['x', 'y', None, 'x', 'y', 'x']})
        seeded()
        gamipy = GAMIPY(data=df, population_size=10, generations=3)
        gamipy.run()
        self.assertFalse(gamipy.imputed_data.isna().any().any())
        self.assertEqual(set(gamipy.imputed_data['b']), {5.0})

    def test_a_column_with_no_observed_value_raises(self):
        gamipy = GAMIPY(data=mixed_frame()[1].assign(empty=np.nan), population_size=10, generations=3)
        with self.assertRaises(ValueError):
            gamipy.run()

    def test_unknown_representation_raises(self):
        with self.assertRaises(ValueError):
            GAMIPY(data=mixed_frame()[1], representation='magic')


if __name__ == '__main__':
    unittest.main()
