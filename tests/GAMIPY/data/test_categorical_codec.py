import unittest

import pandas as pd

from GAMIPY.data.categorical import CategoricalColumn
from GAMIPY.data.categorical_codec import CategoricalCodec
from GAMIPY.data.data import Data


class TestCategoricalCodec(unittest.TestCase):
    def setUp(self):
        data = pd.DataFrame({
            'Category': ['A', 'B', 'A', None, 'C', 'B'],
            'Value': [1, 2, 3, 4, 5, 6]
        })
        self.column = CategoricalColumn(Data(data), 'Category')
        self.codec = CategoricalCodec(self.column.classes)

    def test_encode_maps_each_category_to_its_index_in_classes(self):
        expected = {cls: idx for idx, cls in enumerate(self.column.classes)}
        pd.testing.assert_series_equal(
            self.codec.encode(self.column.data),
            self.column.data.map(expected),
            check_names=False,
        )

    def test_decode_recovers_the_original_category_labels(self):
        decoded = self.codec.decode(self.codec.encode(self.column.data))
        # Missing cells stay missing (NaN doesn't round-trip through the
        # code->label map, since NaN was never assigned a code by encode);
        # compare non-missing values directly and missingness separately to
        # sidestep pandas' NaN-vs-None equality quirk between object columns.
        self.assertTrue(decoded.isna().equals(self.column.data.isna()))
        non_missing = ~self.column.data.isna()
        self.assertEqual(list(decoded[non_missing]), list(self.column.data[non_missing]))

    def test_encode_of_a_value_outside_the_known_classes_becomes_nan(self):
        # Characterization test for a known bug:
        # encode's .map() silently turns any value not in `classes` into NaN
        # rather than raising. This is precisely what made ConstantIndividual/
        # ZerosIndividual/OnesIndividual's old dtype="num" bug (writing 0.0/1.0
        # into a categorical cell) fail silently instead of loudly - see
        # population/individual.py's ConstantIndividual, fixed to use the
        # column's mode instead of an out-of-domain constant.
        encoded = self.codec.encode(pd.Series(['not_a_real_category']))
        self.assertTrue(encoded.isna().all())

    def test_codec_needs_only_the_class_list(self):
        codec = CategoricalCodec(['x', 'y'])
        self.assertEqual(list(codec.encode(pd.Series(['y', 'x']))), [1, 0])
        self.assertEqual(list(codec.decode(pd.Series([0, 1]))), ['x', 'y'])


if __name__ == '__main__':
    unittest.main()
