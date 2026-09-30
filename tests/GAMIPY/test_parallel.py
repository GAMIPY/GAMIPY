"""n_jobs (GAMIPY.parallel): the population is scored in threads, and the
result doesn't depend on the number of threads."""
import os
import random
import unittest

import numpy as np
import pandas as pd

from GAMIPY import GAMIPY
from GAMIPY.parallel import AUTO_THREADS, MIN_CHUNK, chunks, map_chunks, thread_count


def damaged_frame(seed=0, rows=90):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=rows)
    df = pd.DataFrame({
        'a': a,
        'b': 2 * a + rng.normal(scale=0.3, size=rows),
        'c': rng.uniform(0, 5, rows),
        'k': np.where(a > 0, 'hi', 'lo'),
    })
    return df.mask(rng.random(df.shape) < 0.2)


def impute(data, **options):
    random.seed(3)
    np.random.seed(3)
    gamipy = GAMIPY(data=data, population_size=64, generations=8, selection_method='tournament',
                    crossover_rate=0.8, mutation_rate=0.1, **options)
    gamipy.run()
    return gamipy


class TestThreadCount(unittest.TestCase):

    def test_default_is_capped_by_auto_threads_and_cores(self):
        self.assertEqual(thread_count(None), max(1, min(AUTO_THREADS, os.cpu_count() or 1)))

    def test_minus_one_is_every_core(self):
        self.assertEqual(thread_count(-1), os.cpu_count() or 1)

    def test_positive_value_is_used_as_given(self):
        self.assertEqual(thread_count(3), 3)

    def test_invalid_values_raise(self):
        for value in (0, -2):
            with self.subTest(value=value), self.assertRaises(ValueError):
                thread_count(value)

    def test_gamipy_rejects_invalid_n_jobs(self):
        with self.assertRaises(ValueError):
            GAMIPY(data=damaged_frame(), n_jobs=0)


class TestChunks(unittest.TestCase):

    def test_slices_cover_every_item_in_order(self):
        slices = chunks(100, 4)
        self.assertEqual(len(slices), 4)
        covered = [i for part in slices for i in range(100)[part]]
        self.assertEqual(covered, list(range(100)))

    def test_small_totals_are_not_split(self):
        self.assertEqual(chunks(MIN_CHUNK * 2 - 1, 8), [slice(0, MIN_CHUNK * 2 - 1)])

    def test_map_chunks_keeps_slice_order(self):
        results = map_chunks(lambda part: list(range(200)[part]), 200, 4)
        self.assertEqual([i for chunk in results for i in chunk], list(range(200)))


class TestSameResultForAnyThreadCount(unittest.TestCase):

    def assert_same(self, **options):
        data = damaged_frame()
        single = impute(data, n_jobs=1, **options)
        for jobs in (2, 4):
            with self.subTest(n_jobs=jobs):
                threaded = impute(data, n_jobs=jobs, **options)
                pd.testing.assert_frame_equal(single.imputed_data, threaded.imputed_data)
                search, other = single.model or single.runner, threaded.model or threaded.runner
                self.assertEqual(search.best_validation_error, other.best_validation_error)

    def test_model_representation(self):
        self.assert_same()

    def test_cells_representation(self):
        self.assert_same(representation='cells')

    def test_cells_representation_without_standardizing(self):
        self.assert_same(representation='cells', standardize=False)


if __name__ == '__main__':
    unittest.main()
