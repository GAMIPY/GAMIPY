import os

import pandas as pd


class SampleData:
    """
    A class to provide sample datasets for testing and demonstration purposes.

    This class contains methods to load example datasets from CSV files.
    """

    @staticmethod
    def mock_data1() -> pd.DataFrame:
        """
        Load the first example dataset.


        Returns:
            pd.DataFrame: The first example dataset.
        """
        mock_data1 = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [5, 4, 3, 2, 1],
            "C": ["cat", "dog", "cat", None, "cat"],
            "D": [10.5, 20.5, None, 40.5, 50.5]
        })
        return mock_data1

    def load_data(self, name:str) -> pd.DataFrame:
        """
        Load a dataset by name.
        These datasets are used for testing and demonstration purposes.
        These datasets names are:

        - iris
        - ecoli
        - wine
        - wdbc
        - student
        - bank_marketing

        Args:

            - **name (str)**: The name of the dataset to load.

        Returns:

            - **pd.DataFrame**: The loaded dataset.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
        """

        try:
            base_dir = os.path.dirname(__file__)
            file_path = os.path.join(base_dir, name, f"{name}.csv")
            return pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset '{name}' not found.") from None

    def data_info(self, name:str) -> str:
        """
        Load dataset information from a .rst file.
        These datasets are used for testing and demonstration purposes.
        These datasets names are:

        - iris
        - ecoli
        - wine
        - wdbc
        - student
        - bank_marketing

        Args:
            name (str): The name of the dataset for which to load information.


        Returns:
            str: The content of the dataset information file.

        Raises:
            FileNotFoundError: If the dataset information file does not exist.
        """
        file_path = os.path.join(os.path.dirname(__file__), name,f"{name}.rst")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset info for '{name}' not found.")
        with open(file_path, encoding="utf-8") as file:
            return file.read()




