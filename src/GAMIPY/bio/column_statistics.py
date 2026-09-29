from abc import ABC, abstractmethod
from typing import Any

from GAMIPY.data.categorical import CategoricalColumn
from GAMIPY.data.data import Data
from GAMIPY.data.numerical import NumericalColumn


class ColumnStatisticsProvider(ABC):

    """
    Abstraction for "what's the representative value of this column?" -
    Fitness.mean_score used to answer that by instantiating
    NumericalColumn/CategoricalColumn directly, so the high-level scoring
    logic depended on those two concrete data-layer classes instead of an
    abstraction. Each concrete provider here wraps one
    such class; Fitness depends only on this interface.
    """

    @abstractmethod
    def central_value(self, data: Data, column: str) -> Any:

        """
        Returns the representative value of `column` in `data`.

        Args
        ----
        - **data (Data)**: The dataset containing the column.
        - **column (str)**: The name of the column.

        Returns
        -------
        The column's representative value (a float for a numerical
        provider, a category label for a categorical one).
        """

        pass


class NumericalCentralTendency(ColumnStatisticsProvider):

    """Represents a numerical column by its mean."""

    def central_value(self, data: Data, column: str) -> Any:
        return NumericalColumn(data, column).mean_val


class CategoricalCentralTendency(ColumnStatisticsProvider):

    """Represents a categorical column by its mode (most frequent category)."""

    def central_value(self, data: Data, column: str) -> Any:
        return CategoricalColumn(data, column).mode
