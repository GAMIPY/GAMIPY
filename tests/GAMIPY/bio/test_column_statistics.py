import unittest

import pandas as pd

from GAMIPY.bio.column_statistics import (
    CategoricalCentralTendency,
    NumericalCentralTendency,
)
from GAMIPY.data.data import Data


class TestColumnStatistics(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0],
            'B': ['x', 'x', 'y', 'x'],
        }))

    def test_numerical_central_tendency_returns_the_mean(self):
        value = NumericalCentralTendency().central_value(self.data, 'A')
        self.assertAlmostEqual(value, 2.5)

    def test_categorical_central_tendency_returns_the_mode(self):
        value = CategoricalCentralTendency().central_value(self.data, 'B')
        self.assertEqual(value, 'x')
