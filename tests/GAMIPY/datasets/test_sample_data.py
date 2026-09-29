import unittest

import pandas as pd

from GAMIPY.datasets.sample_data import SampleData


class TestMockData1(unittest.TestCase):
    def test_returns_expected_shape_and_columns(self):
        df = SampleData.mock_data1()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (5, 4))
        self.assertEqual(list(df.columns), ["A", "B", "C", "D"])

    def test_contains_missing_values_in_c_and_d(self):
        df = SampleData.mock_data1()
        self.assertEqual(df["C"].isna().sum(), 1)
        self.assertEqual(df["D"].isna().sum(), 1)
        self.assertEqual(df["A"].isna().sum(), 0)
        self.assertEqual(df["B"].isna().sum(), 0)


class TestLoadData(unittest.TestCase):
    def setUp(self):
        self.sample_data = SampleData()

    def test_iris_has_expected_shape_and_columns(self):
        df = self.sample_data.load_data("iris")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (150, 5))
        self.assertEqual(
            list(df.columns),
            ['sepal length', 'sepal width', 'petal length', 'petal width', 'class'],
        )

    def test_every_documented_dataset_name_loads_a_non_empty_dataframe(self):
        # Names as documented in SampleData.load_data's own docstring.
        for name in ["iris", "ecoli", "wine", "wdbc", "student", "bank_marketing"]:
            with self.subTest(name=name):
                df = self.sample_data.load_data(name)
                self.assertIsInstance(df, pd.DataFrame)
                self.assertFalse(df.empty)

    def test_invalid_name_raises_file_not_found_error(self):
        with self.assertRaises(FileNotFoundError):
            self.sample_data.load_data("not_a_real_dataset")


class TestDataInfo(unittest.TestCase):
    def setUp(self):
        self.sample_data = SampleData()

    def test_wdbc_returns_non_empty_string(self):
        info = self.sample_data.data_info("wdbc")
        self.assertIsInstance(info, str)
        self.assertGreater(len(info), 0)

    def test_every_dataset_has_a_description(self):
        # Every name load_data() accepts - including bank_marketing, whose
        # description used to be a .txt file, so data_info() raised
        # FileNotFoundError for it and the wheel didn't ship it.
        for name in ["iris", "ecoli", "wine", "wdbc", "student", "bank_marketing"]:
            with self.subTest(name=name):
                info = self.sample_data.data_info(name)
                self.assertIsInstance(info, str)
                self.assertGreater(len(info), 100)
                self.assertFalse(info.startswith("\ufeff"))  # no byte-order mark

    def test_invalid_name_raises_file_not_found_error(self):
        with self.assertRaises(FileNotFoundError):
            self.sample_data.data_info("not_a_real_dataset")


class TestMockDataCsv(unittest.TestCase):
    """examples/mock_data.csv is a readable dump of SampleData.mock_data1(),
    the single source of truth - it used to be copied to four places that
    could silently drift apart."""

    def test_examples_csv_matches_mock_data1(self):
        import os
        path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples', 'mock_data.csv')
        expected = SampleData.mock_data1()
        # read_csv yields NaN for missing cells where mock_data1 has None;
        # pandas is deprecating treating those two as equal.
        expected = expected.where(expected.notna(), float('nan'))
        pd.testing.assert_frame_equal(pd.read_csv(path), expected, check_dtype=False)


if __name__ == '__main__':
    unittest.main()
