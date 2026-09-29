import unittest

import pandas as pd

from GAMIPY.bio.value_sampling import (
    CategoricalValueSampler,
    NumericalValueSampler,
)
from GAMIPY.data.data import Data


class TestValueSampling(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0],
            'B': ['x', 'y', 'z', 'x'],
        }))

    def test_numerical_value_sampler_stays_within_the_column_range(self):
        sampler = NumericalValueSampler()
        for _ in range(20):
            value = sampler.sample(self.data, 'A')
            self.assertGreaterEqual(value, 1.0)
            self.assertLessEqual(value, 4.0)

    def test_categorical_value_sampler_returns_a_known_category(self):
        sampler = CategoricalValueSampler()
        for _ in range(20):
            value = sampler.sample(self.data, 'B')
            self.assertIn(value, ['x', 'y', 'z'])
