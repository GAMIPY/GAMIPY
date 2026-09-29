from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


class Data:

    """
    A class to represent a dataset and provide various utility methods for data analysis.

    Attributes
    ----------
    - **data (pd.DataFrame)**: The dataset as a pandas DataFrame.
    - **get_numeric_columns (list)**: List of numeric column names.
    - **get_categorical_columns (list)**: List of categorical column names.
    - **get_columns (list)**: List of all column names.
    - **get_n_rows (int)**: Number of rows in the dataset.
    - **get_n_cols (int)**: Number of columns in the dataset.
    - **get_data (pd.DataFrame)**: The dataset as a pandas DataFrame.
    - **get_missing_column_names (list)**: List of column names that contain missing values.
    - **get_complete_data (pd.DataFrame)**: DataFrame with complete cases (no missing values).
    - **get_missing_data (pd.DataFrame)**: DataFrame with rows that contain missing values.
    - **to_dataframe (pd.DataFrame)**: Converts the data object back to a pandas DataFrame.

    Methods
    -------
    - **get_column(column_name)**: Returns a specific column as a pandas Series.
    """

    def __init__(self, data: pd.DataFrame) -> None:

        """
        Initializes the Data object with a pandas DataFrame.

        Args
        ----
        - **data (pd.DataFrame)**: The dataset to be analyzed.

        Raises
        ------
        - **ValueError**: If the input data is not a pandas DataFrame.
        """

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        self.data = data



    @property
    def get_numeric_columns(self) -> list[Any]:

        """
        Returns a list of numeric column names in the dataset.

        Returns
        -------
            - **list**: A list of numeric column names.

        """

        return list(self._column_types()[0])

    @property
    def get_categorical_columns(self) -> list[Any]:

        """
        Returns a list of categorical column names in the dataset.

        Returns
        -------
        - **list**: A list of categorical column names.

        """

        return list(self._column_types()[1])

    def column_profile(self, column: Any) -> ColumnProfile:
        """
        What initialization samples from for `column`: the number of observed
        and of distinct observed values, the observed min/max (numerical
        columns), the distinct labels in order of appearance and the most
        frequent one - the same values NumericalColumn/CategoricalColumn
        compute. Cached per column until the DataFrame changes, because every
        gene of every new individual asks for it.
        """
        cache: dict[Any, Any] = self.__dict__.setdefault('_profile_cache', {})
        key = (id(self.data), column)
        if key not in cache:
            observed = self.data[column].dropna()
            numeric = column in self._column_types()[0]
            modes = observed.mode()
            cache[key] = ColumnProfile(
                observed=len(observed),
                distinct=int(observed.nunique()),
                minimum=observed.min() if numeric and len(observed) else None,
                maximum=observed.max() if numeric and len(observed) else None,
                classes=observed.unique().tolist(),
                mode=modes.iloc[0] if not modes.empty else None,
            )
        profile: ColumnProfile = cache[key]
        return profile

    def _column_types(self) -> tuple[list[Any], list[Any]]:
        """(numeric columns, categorical columns), cached until the DataFrame
        or its dtypes change: both lists are asked for once per gene and per
        fitness evaluation, and select_dtypes is comparatively slow."""
        dtypes = self.data.dtypes
        key = (id(self.data), tuple(dtypes.items()))
        cache = getattr(self, '_column_types_cache', None)
        if cache is None or cache[0] != key:
            numeric = self.data.select_dtypes(include=['number']).columns.tolist()
            # Checked per dtype instead of select_dtypes(include=['object', ...]):
            # pandas 3 gives text columns the new 'str' dtype (a StringDtype),
            # which 'object' only still matches through a deprecated fallback
            # (Pandas4Warning), while pandas 2 rejects 'str' in select_dtypes.
            categorical = [
                col for col, dtype in dtypes.items()
                if pd.api.types.is_object_dtype(dtype)
                or isinstance(dtype, (pd.StringDtype, pd.CategoricalDtype))
            ]
            cache = (key, (numeric, categorical))
            self._column_types_cache = cache
        return cache[1]

    @property
    def get_columns(self) -> list[Any]:

        """
        Returns a list of all column names in the dataset.

        Returns
        -------
        - **list**: A list of all column names in the dataset.
        """

        return self.data.columns.tolist()

    @property
    def get_n_rows(self) -> int:

        """
        Returns the number of rows in the dataset.

        Returns
        -------
        - **int**: The number of rows in the dataset.
        """

        return self.data.shape[0]

    @property
    def get_n_cols(self) -> int:
        """
        Returns the number of columns in the dataset.

        Returns
        -------
        - **int**: The number of columns in the dataset.
        """
        return self.data.shape[1]

    @property
    def get_data(self) -> pd.DataFrame:
        """
        Returns the dataset as a pandas DataFrame.

        Returns
        -------
        - **pd.DataFrame**: The dataset as a pandas DataFrame.
        """
        return self.data

    @property
    def get_missing_column_names(self) -> list[Any]:
        """
        Returns a list of column names that contain missing values.

        Returns
        -------
        - **list**: A list of column names with missing values.

        Raises
        ------
        - **ValueError**: If no columns with missing values are found in the dataset.
        """

        return self.data.columns[self.data.isnull().any()].tolist()

    @property
    def get_complete_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame with complete cases (no missing values).

        Returns
        -------
        - **pd.DataFrame**: A DataFrame containing only complete cases (no missing values).
        """

        return self.data.dropna()

    @property
    def get_missing_data(self) -> pd.DataFrame:

        """
        Returns a DataFrame with rows that contain missing values.

        Returns
        -------
        - **pd.DataFrame**: A DataFrame containing rows with missing values.
        """

        return self.data[self.data.isnull().any(axis=1)].reset_index(drop=True)

    @property
    def to_dataframe(self) -> pd.DataFrame:

        """
        Returns a copy of the dataset as a pandas DataFrame.

        Returns
        -------
        - **pd.DataFrame**: A copy of the dataset as a pandas DataFrame.
        """

        return self.data.copy()


    def get_column(self, column_name: str) -> pd.Series[Any]:

        """
        Returns a specific column from the dataset as a pandas Series.

        Args
        ----
        - **column_name (str)**: The name of the column to retrieve.

        Returns
        -------
        - **pd.Series**: The specified column as a pandas Series.

        Raises
        ------
        - **ValueError**: If the specified column does not exist in the dataset.
        """

        if column_name not in self.get_columns:
            raise ValueError(f"Column '{column_name}' does not exist in the data.")


        return self.data[column_name]

    def __repr__(self) -> str:

        """
        Returns a string representation of the Data object.

        Returns
        -------
        - **str**: A string representation of the Data object.
        """

        return f"Data(n_rows={self.get_n_rows}, n_cols={self.get_n_cols}, columns={self.get_columns})"


@dataclass(frozen=True)
class ColumnProfile:
    """Observed-value summary of one column, see ``Data.column_profile``."""

    observed: int
    distinct: int
    minimum: Any
    maximum: Any
    classes: list[Any]
    mode: Any
