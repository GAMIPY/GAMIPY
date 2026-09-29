from functools import cached_property
from typing import Any

from GAMIPY.data.column import Column
from GAMIPY.data.data import Data


class NumericalColumn(Column):

    """
    Represents a numerical column in a dataset.
    This class extends the Column class to provide additional functionality specific to numerical data.
    It includes methods for calculating various statistical properties of the column, such as mean,
    median, variance, and skewness.

    Attributes
    ----------
    - **min_val (float)**: The minimum value in the column.
    - **max_val (float)**: The maximum value in the column.
    - **mean_val (float)**: The mean (average) value of the column.
    - **std_val (float)**: The standard deviation of the column.
    - **median_val (float)**: The median value of the column.
    - **variance_val (float)**: The variance of the column.
    - **range_val (float)**: The range of the column (max - min).
    - **skewness_val (float)**: The skewness of the column, indicating the asymmetry of the distribution.
    - **kurtosis_val (float)**: The kurtosis of the column, indicating the "tailedness" of the distribution.
    """

    def __init__(self, dataset: Data, column_name: str = "") -> None:

        """
        Initializes the NumericalColumn object with a dataset and the name of the numerical column.

        Args
        ----
        - **data (Data)**: The dataset containing the numerical column.
        - **column_name (str)**: The name of the numerical column to be accessed.

        Raises
        ------
        - **ValueError**: If the specified column is not numeric or does not exist in the dataset.
        """

        super().__init__(dataset, column_name)
        if column_name not in dataset.get_numeric_columns:
            raise ValueError(f"Column '{column_name}' is not numeric.")

    # Each statistic is computed on first access and then cached: callers
    # typically need one or two of them (min/max to sample a value, the mean
    # for fitness), and computing all nine eagerly for every column object
    # dominated the GA's run time.
    @cached_property
    def min_val(self) -> Any:
        return self.data.min()

    @cached_property
    def max_val(self) -> Any:
        return self.data.max()

    @cached_property
    def mean_val(self) -> Any:
        return self.data.mean()

    @cached_property
    def std_val(self) -> Any:
        return self.data.std()

    @cached_property
    def median_val(self) -> Any:
        return self.data.median()

    @cached_property
    def variance_val(self) -> Any:
        return self.data.var()

    @cached_property
    def range_val(self) -> Any:
        return self.max_val - self.min_val

    @cached_property
    def skewness_val(self) -> Any:
        return self.data.skew()

    @cached_property
    def kurtosis_val(self) -> Any:
        return self.data.kurtosis()

    @property
    def summary(self) -> dict[str, Any]:

        """
        Returns a summary of the numerical column, including its statistical properties.

        Returns
        -------
        - **dict**: A dictionary containing the column name, number of rows, number of missing
          values, and various statistical properties such as min, max, mean, std, median, variance,
          range, skewness, and kurtosis.
        """


        base_summary = super().summary
        return {
            **base_summary,
            "min": self.min_val,
            "max": self.max_val,
            "mean": self.mean_val,
            "std": self.std_val,
            "median": self.median_val,
            "variance": self.variance_val,
            "range": self.range_val,
            "skewness": self.skewness_val,
            "kurtosis": self.kurtosis_val
        }
