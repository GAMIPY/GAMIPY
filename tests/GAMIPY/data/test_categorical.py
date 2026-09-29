import unittest

import pandas as pd

from GAMIPY.data.categorical import CategoricalColumn
from GAMIPY.data.data import Data


class TestCategoricalColumn(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset with categorical data
        data = pd.DataFrame({
            'Category': ['A', 'B', 'A', None, 'C', 'B'],
            'Value': [1, 2, 3, 4, 5, 6]
        })
        self.data = Data(data)

    def test_categorical_column_initialization(self):
        column = CategoricalColumn(self.data, 'Category')
        self.assertEqual(column.name, 'Category')
        self.assertEqual(column.n_rows, 6)
        self.assertEqual(column.n_missing, 1)

    def test_classes_excludes_the_missing_value_itself(self):
        # .unique() alone would include NaN as a "class" whenever the column
        # has missing values, so random.choice(cat_col.classes) could pick
        # NaN as if it were a real category (RandomIndividual.individual(),
        # Mutation.mutate() both rely on .classes for exactly this).
        column = CategoricalColumn(self.data, 'Category')
        self.assertEqual(sorted(column.classes), ['A', 'B', 'C'])
        self.assertEqual(column.n_classes, 3)
        self.assertNotIn(None, column.classes)
        self.assertFalse(any(pd.isna(c) for c in column.classes))

    def test_invalid_categorical_column_name(self):
        with self.assertRaises(ValueError):
            CategoricalColumn(self.data, 'NonExistent')

    def test_no_missing_values_categorical_column(self):
        with self.assertRaises(ValueError):
            CategoricalColumn(self.data, 'Value')

    def test_summary_only_client_does_not_get_encode_decode(self):
        # ISP: encode/decode live on CategoricalCodec, not on the column.
        column = CategoricalColumn(self.data, 'Category')
        self.assertFalse(hasattr(column, 'encode'))
        self.assertFalse(hasattr(column, 'decode'))

    def test_mode_ignores_missing_values(self):
        # Found by tests/GAMIPY/test_integration.py: missing cells used to
        # be counted as the category 'None', which could win the vote.
        data = Data(pd.DataFrame({'Category': [None, None, 'A'], 'Value': [1, 2, 3]}))
        column = CategoricalColumn(data, 'Category')
        self.assertEqual(column.mode, 'A')
        self.assertIn(column.mode, column.classes)

    def test_mode_of_an_all_missing_column_is_none(self):
        data = Data(pd.DataFrame({'Category': pd.Series([None, None], dtype=object), 'Value': [1, 2]}))
        self.assertIsNone(CategoricalColumn(data, 'Category').mode)

    def test_categorical_summary(self):
        column = CategoricalColumn(self.data, 'Category')
        summary = column.summary
        self.assertEqual(summary['column_name'], 'Category')
        self.assertEqual(summary['n_rows'], 6)
        self.assertEqual(summary['n_missing'], 1)
