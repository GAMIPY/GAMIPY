import unittest

import pandas as pd

from GAMIPY.data.data import Data
from GAMIPY.data.numerical import NumericalColumn


class TestNumericalColumn(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset with numerical data
        data = pd.DataFrame({
            'A': [1.0, 2.5, None, 4.0],
            'B': [3.0, 5.5, 6.0, 7.0],
            'C': [8.0, None, None, 10.0]
        })
        self.data = Data(data)

    def test_numerical_column_initialization(self):
        column = NumericalColumn(self.data, 'A')
        self.assertEqual(column.name, 'A')
        self.assertEqual(column.n_rows, 4)
        self.assertEqual(column.n_missing, 1)

    def test_invalid_numerical_column_name(self):
        with self.assertRaises(ValueError):
            NumericalColumn(self.data, 'D')

    def test_numerical_summary(self):
        column = NumericalColumn(self.data, 'C')
        summary = column.summary
        self.assertEqual(summary['column_name'], 'C')
        self.assertEqual(summary['n_rows'], 4)
        self.assertEqual(summary['n_missing'], 2)
        self.assertEqual(
            set(summary),
            {'column_name', 'n_rows', 'n_missing', 'min', 'max', 'mean', 'std',
             'median', 'variance', 'range', 'skewness', 'kurtosis'},
        )

    def test_summary_values_on_a_complete_column(self):
        # The summary used to be checked for key presence
        # only. B = [3.0, 5.5, 6.0, 7.0]; expected values computed by hand
        # (sample statistics, ddof=1), skewness/kurtosis from SciPy's
        # bias-corrected formulas - an implementation independent of pandas.
        from scipy import stats
        values = [3.0, 5.5, 6.0, 7.0]
        summary = NumericalColumn(self.data, 'B').summary
        self.assertEqual(summary['n_missing'], 0)
        self.assertEqual(summary['min'], 3.0)
        self.assertEqual(summary['max'], 7.0)
        self.assertEqual(summary['range'], 4.0)
        self.assertAlmostEqual(summary['mean'], 5.375)
        self.assertAlmostEqual(summary['median'], 5.75)
        self.assertAlmostEqual(summary['variance'], 8.6875 / 3)
        self.assertAlmostEqual(summary['std'], (8.6875 / 3) ** 0.5)
        self.assertAlmostEqual(summary['skewness'], stats.skew(values, bias=False))
        self.assertAlmostEqual(summary['kurtosis'], stats.kurtosis(values, bias=False))

    def test_summary_ignores_missing_values(self):
        # A = [1.0, 2.5, NaN, 4.0]: statistics over the 3 observed values.
        summary = NumericalColumn(self.data, 'A').summary
        self.assertEqual(summary['n_missing'], 1)
        self.assertEqual((summary['min'], summary['max'], summary['range']), (1.0, 4.0, 3.0))
        self.assertAlmostEqual(summary['mean'], 2.5)
        self.assertAlmostEqual(summary['median'], 2.5)
        self.assertAlmostEqual(summary['variance'], 2.25)   # ((-1.5)^2 + 0 + 1.5^2) / 2

    def test_skewness_and_kurtosis_are_nan_with_too_few_values(self):
        # C has 2 observed values; skewness needs 3 and kurtosis 4.
        summary = NumericalColumn(self.data, 'C').summary
        self.assertAlmostEqual(summary['mean'], 9.0)
        self.assertTrue(pd.isna(summary['skewness']))
        self.assertTrue(pd.isna(summary['kurtosis']))
