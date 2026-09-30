"""The per-generation ``history`` of ModelSearch and GeneticAlgorithmRunner."""
import math
import random
import unittest

import numpy as np
import pandas as pd

from GAMIPY import GAMIPY


def damaged_frame(seed=0, rows=60):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=rows)
    df = pd.DataFrame({'a': a, 'b': a + rng.normal(scale=0.3, size=rows), 'c': rng.uniform(0, 5, rows)})
    return df.mask(rng.random(df.shape) < 0.2)


def run(**options):
    random.seed(0)
    np.random.seed(0)
    gamipy = GAMIPY(data=damaged_frame(), population_size=20, generations=6, selection_method='tournament',
                    crossover_rate=0.8, mutation_rate=0.1, patience=0, **options)
    gamipy.run()
    return gamipy.model or gamipy.runner


class TestHistory(unittest.TestCase):

    def check(self, search, score_key):
        self.assertEqual(len(search.history), search.generations_run)
        self.assertEqual([entry['generation'] for entry in search.history],
                         list(range(1, search.generations_run + 1)))
        validation = [entry['best_validation_error'] for entry in search.history]
        # Best-so-far values never get worse.
        self.assertEqual(validation, sorted(validation, reverse=True))
        self.assertAlmostEqual(validation[-1], search.best_validation_error)
        self.assertTrue(all(math.isfinite(entry[score_key]) for entry in search.history))

    def test_model_representation(self):
        self.check(run(), 'best_training_error')

    def test_cells_representation(self):
        self.check(run(representation='cells'), 'best_score')

    def test_without_validation_the_validation_error_is_nan(self):
        search = run(representation='cells', validation_fraction=0.0)
        self.assertEqual(len(search.history), 6)
        self.assertTrue(all(math.isnan(entry['best_validation_error']) for entry in search.history))


if __name__ == '__main__':
    unittest.main()
