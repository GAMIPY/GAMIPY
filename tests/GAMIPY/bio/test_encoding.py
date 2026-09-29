import unittest

import pandas as pd

from GAMIPY.bio.encoding import CategoricalEncoder
from GAMIPY.data.data import Data


class TestCategoricalEncoder(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0],
            'B': ['x', 'y', 'x', 'y'],
        })
        self.data = Data(self.df)
        self.encoder = CategoricalEncoder()

    def test_encode_replaces_categorical_column_with_integer_codes(self):
        encoded = self.encoder.encode(self.data)
        self.assertTrue(pd.api.types.is_numeric_dtype(encoded.data['B']))
        # Numeric column is left untouched.
        pd.testing.assert_series_equal(encoded.data['A'], self.df['A'], check_names=False)

    def test_decode_recovers_original_category_labels(self):
        encoded = self.encoder.encode(self.data)
        decoded = self.encoder.decode(self.data, encoded.data)
        pd.testing.assert_series_equal(decoded.data['B'], self.df['B'], check_names=False)

    def test_encode_decode_round_trip_is_identity(self):
        encoded = self.encoder.encode(self.data)
        decoded = self.encoder.decode(self.data, encoded.data)
        pd.testing.assert_frame_equal(decoded.data, self.df)
