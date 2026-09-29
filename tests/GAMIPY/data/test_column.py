import unittest

import pandas as pd

from GAMIPY.data.column import Column
from GAMIPY.data.data import Data


class TestColumn(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset with missing values
        data = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [3, 5, 6, 7],
            'C': [8, None, None, 10]
        })
        self.data = Data(data)


    def test_column_initialization(self):
        column = Column(self.data, 'A')
        self.assertEqual(column.name, 'A')
        self.assertEqual(column.n_rows, 4)
        self.assertEqual(column.n_missing, 1)

    def test_invalid_column_name(self):
        with self.assertRaises(ValueError):
            Column(self.data, 'D')


    def test_summary(self):
        column = Column(self.data, 'C')
        summary = column.summary
        self.assertEqual(summary['column_name'], 'C')
        self.assertEqual(summary['n_rows'], 4)
        self.assertEqual(summary['n_missing'], 2)
