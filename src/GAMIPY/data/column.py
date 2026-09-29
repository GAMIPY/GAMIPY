from typing import Any

from GAMIPY.data.data import Data


class Column:

    """
    Represents a single column in a dataset. This class provides access to the data in a specific
    column, along with metadata such as the number of rows and missing values. It is initialized
    with a dataset and the name of the column to be accessed.

    Attributes
    ----------
    - **dataset (Data)**: The dataset containing the column.
    - **name (str)**: The name of the column.
    - **data (pd.Series)**: The data in the column as a pandas Series
    - **n_rows (int)**: The number of rows in the dataset.
    - **n_missing (int)**: The number of missing values in the column.
    """

    def __init__(self, dataset: Data, column_name: str) -> None:

        """
        Initializes the Column object with a dataset and the name of the column to be accessed.

        Args
        ----
        - **dataset (Data)**: The dataset containing the column.
        - **column_name (str)**: The name of the column to be accessed.

        Raises
        ------
        - **ValueError**: If the specified column does not exist in the dataset.
        """

        self.dataset = dataset
        self.name = column_name
        if column_name not in dataset.get_columns:
            raise ValueError(f"Column '{column_name}' does not exist in the data.")
        #if column_name not in dataset.get_missing_column_names():
        #    raise ValueError(f"Column '{column_name}' does not have missing values.")
        self.data = dataset.get_column(column_name)

        self.n_rows = dataset.get_n_rows
        self.n_missing = self.data.isnull().sum()

    @property
    def summary(self) -> dict[str, Any]:

        """
        Returns a summary of the column, including its name, number of rows, and number of missing values.

        Returns
        -------
        - **dict**: A dictionary containing the column name, number of rows, and number of missing values.
        """

        return {
            "column_name": self.name,
            "n_rows": self.n_rows,
            "n_missing": self.n_missing
        }
