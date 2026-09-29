import unittest

import pandas as pd

from GAMIPY.data.data import Data


class TestData(unittest.TestCase):

    def setUp(self):

        self.data = pd.DataFrame({
            'A': [1, 2, None],
            'B': ['x', 'y', 'z'],
            'C': [3.5, None, 4.5]
        })
        self.data_instance = Data(self.data)

    def test_get_numeric_columns(self):
        numeric_columns = self.data_instance.get_numeric_columns
        self.assertEqual(numeric_columns, ['A', 'C'])
    def test_get_categorical_columns(self):
        categorical_columns = self.data_instance.get_categorical_columns
        self.assertEqual(categorical_columns, ['B'])
    def test_categorical_columns_cover_every_text_dtype(self):
        df = pd.DataFrame({
            'obj': pd.Series(['a', 'b'], dtype=object),
            'string': pd.Series(['a', 'b'], dtype='string'),
            'cat': pd.Series(['a', 'b'], dtype='category'),
            'num': [1.0, 2.0],
        })
        self.assertEqual(Data(df).get_categorical_columns, ['obj', 'string', 'cat'])
    def test_get_columns(self):
        columns = self.data_instance.get_columns
        self.assertEqual(columns, ['A', 'B', 'C'])
    def test_get_n_rows(self):
        n_rows = self.data_instance.get_n_rows
        self.assertEqual(n_rows, 3)
    def test_get_n_cols(self):
        n_cols = self.data_instance.get_n_cols
        self.assertEqual(n_cols, 3)
    def test_get_data(self):
        data = self.data_instance.get_data
        pd.testing.assert_frame_equal(data, self.data)
    def test_get_column(self):
        column_a = self.data_instance.get_column('A')
        pd.testing.assert_series_equal(column_a, self.data['A'])
        with self.assertRaises(ValueError):
            self.data_instance.get_column('D')
    def test_get_missing_column_names(self):
        missing_columns = self.data_instance.get_missing_column_names
        self.assertEqual(missing_columns, ['A', 'C'])
    def test_get_complete_data(self):
        complete_data = self.data_instance.get_complete_data
        expected_complete = pd.DataFrame({
                'A': [1.0],
                'B': ['x'],
                'C': [3.5]
            }, index=[0])
        pd.testing.assert_frame_equal(complete_data, expected_complete)
    def test_get_missing_data(self):
        missing_data = self.data_instance.get_missing_data
        expected_missing = pd.DataFrame({
            'A': [2.0, None],
            'B': ['y', 'z'],
            'C': [None, 4.5]
        }, index=[1, 2]).reset_index(drop=True)
        pd.testing.assert_frame_equal(missing_data, expected_missing)
    def test_data_no_longer_computes_missing_index_itself(self):
        # SRP: moved to data/missing_index.MissingIndexLocator.
        self.assertFalse(hasattr(self.data_instance, 'get_missing_index'))
    def test_to_dataframe(self):
        df_copy = self.data_instance.to_dataframe
        pd.testing.assert_frame_equal(df_copy, self.data)
        self.assertIsNot(df_copy, self.data)
if __name__ == '__main__':
    unittest.main()

