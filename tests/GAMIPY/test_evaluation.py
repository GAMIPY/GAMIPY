import unittest

import numpy as np
import pandas as pd

from GAMIPY.evaluation import column_errors, distribution_scores, imputation_scores


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.original = pd.DataFrame({
            'x': [1.0, 2.0, 3.0, 5.0],
            'y': [10.0, 20.0, 30.0, 40.0],
            'c': ['a', 'b', 'a', 'b'],
        })
        self.damaged = self.original.copy()
        self.damaged.loc[[0, 1], 'x'] = np.nan
        self.damaged.loc[2, 'y'] = np.nan
        self.damaged.loc[[1, 3], 'c'] = None
        self.imputed = self.original.copy()
        self.imputed.loc[0, 'x'] = 4.0     # error +3
        self.imputed.loc[1, 'x'] = 1.0     # error -1
        self.imputed.loc[2, 'y'] = 26.0    # error -4
        self.imputed.loc[1, 'c'] = 'a'     # wrong
        self.imputed.loc[3, 'x'] = 100.0   # observed cell: must be ignored

    def test_column_errors_use_hidden_cells_only(self):
        errors = column_errors(self.original, self.damaged, self.imputed)
        self.assertEqual(list(errors.index), ['x', 'y', 'c'])
        self.assertEqual(errors.loc['x', 'n'], 2)
        self.assertAlmostEqual(errors.loc['x', 'RMSE'], np.sqrt((9 + 1) / 2))
        self.assertAlmostEqual(errors.loc['x', 'MAE'], 2.0)
        self.assertAlmostEqual(errors.loc['x', 'NRMSE'], np.sqrt(5) / 4.0)   # range 1..5
        self.assertAlmostEqual(errors.loc['y', 'RMSE'], 4.0)
        self.assertAlmostEqual(errors.loc['c', 'accuracy'], 0.5)

    def test_scores_pool_over_cells(self):
        scores = imputation_scores(self.original, self.damaged, self.imputed)
        self.assertEqual(scores['n_numerical'], 3)
        self.assertEqual(scores['n_categorical'], 2)
        self.assertAlmostEqual(scores['RMSE'], np.sqrt((9 + 1 + 16) / 3))
        self.assertAlmostEqual(scores['MAE'], (3 + 1 + 4) / 3)
        self.assertAlmostEqual(scores['accuracy'], 0.5)

    def test_unfilled_hidden_cell_raises_instead_of_being_skipped(self):
        imputed = self.imputed.copy()
        imputed.loc[0, 'x'] = np.nan
        with self.assertRaises(ValueError):
            column_errors(self.original, self.damaged, imputed)

    def test_misaligned_frames_raise(self):
        with self.assertRaises(ValueError):
            column_errors(self.original, self.damaged, self.imputed.iloc[::-1])
        with self.assertRaises(ValueError):
            column_errors(self.original, self.damaged[['x', 'y']], self.imputed)

    def test_nothing_hidden_gives_empty_errors_and_nan_scores(self):
        errors = column_errors(self.original, self.original, self.original)
        self.assertTrue(errors.empty)
        scores = imputation_scores(self.original, self.original, self.original)
        self.assertTrue(np.isnan(scores['RMSE']) and np.isnan(scores['accuracy']))



class TestDistributionScores(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=200)
        self.original = pd.DataFrame({
            'a': a, 'b': a + rng.normal(scale=0.5, size=200), 'c': rng.choice(['x', 'y'], 200),
        })
        self.damaged = self.original.copy()
        self.damaged.loc[rng.random(200) < 0.3, 'b'] = np.nan

    def test_the_truth_preserves_everything(self):
        scores = distribution_scores(self.original, self.damaged, self.original)
        self.assertAlmostEqual(scores['variance_ratio'], 1.0)
        self.assertAlmostEqual(scores['correlation_error'], 0.0)
        self.assertAlmostEqual(scores['KS'], 0.0)

    def test_mean_imputation_shrinks_variance_and_correlation(self):
        imputed = self.damaged.fillna({'b': self.damaged['b'].mean()})
        scores = distribution_scores(self.original, self.damaged, imputed)
        self.assertLess(scores['variance_ratio'], 0.9)
        self.assertGreater(scores['correlation_error'], 0.05)
        self.assertGreater(scores['KS'], 0.3)

    def test_observed_cells_of_the_imputed_frame_are_ignored(self):
        imputed = self.original.copy()
        imputed.loc[self.damaged['b'].notna(), 'b'] = 1e6
        self.assertAlmostEqual(distribution_scores(self.original, self.damaged, imputed)['variance_ratio'], 1.0)


if __name__ == '__main__':
    unittest.main()
