import unittest

import pandas as pd

from GAMIPY.data.data import Data
from GAMIPY.data.missing_index import MissingIndexLocator


class TestMissingIndexLocator(unittest.TestCase):

    def test_locate_returns_original_row_missing_row_position_and_column(self):
        data = Data(pd.DataFrame({
            'A': [1, 2, None],
            'B': ['x', 'y', 'z'],
            'C': [3.5, None, 4.5]
        }))
        self.assertEqual(MissingIndexLocator(data).locate(), [(1, 0, 'C'), (2, 1, 'A')])

    def test_row_m_matches_position_in_get_missing_data(self):
        df = pd.DataFrame({
            'f1': [1.0, None, 3.0, None, 5.0],
            'f2': [None, 7.0, 8.0, 9.0, None],
        }, index=[10, 11, 13, 14, 15])
        data = Data(df)
        m_data = data.get_missing_data
        for row, row_m, col in MissingIndexLocator(data).locate():
            self.assertTrue(pd.isna(df.loc[row, col]))
            self.assertTrue(pd.isna(m_data.loc[row_m, col]))
            pd.testing.assert_series_equal(
                m_data.loc[row_m], df.loc[row], check_names=False
            )

    def test_several_missing_cells_in_one_row_share_its_row_m(self):
        data = Data(pd.DataFrame({'A': [1.0, None], 'B': [2.0, None]}))
        self.assertEqual(MissingIndexLocator(data).locate(), [(1, 0, 'A'), (1, 0, 'B')])

    def test_duplicate_index_labels_get_distinct_row_m(self):
        # Rows are walked by position, not label.
        data = Data(pd.DataFrame({'A': [None, None, 3.0]}, index=[7, 7, 8]))
        self.assertEqual(MissingIndexLocator(data).locate(), [(7, 0, 'A'), (7, 1, 'A')])

    def test_complete_data_has_no_missing_locations(self):
        data = Data(pd.DataFrame({'A': [1.0, 2.0]}))
        self.assertEqual(MissingIndexLocator(data).locate(), [])


if __name__ == '__main__':
    unittest.main()
