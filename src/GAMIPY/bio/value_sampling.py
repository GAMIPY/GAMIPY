import random
from abc import ABC, abstractmethod
from typing import Any

from GAMIPY.data.categorical import CategoricalColumn
from GAMIPY.data.data import Data
from GAMIPY.data.numerical import NumericalColumn


class ValueSampler(ABC):

    """
    Abstraction for "give me a plausible random value for this column" -
    Mutation.mutate used to answer that by instantiating
    NumericalColumn/CategoricalColumn directly, so mutation depended on
    those two concrete data-layer classes instead of an abstraction. Each concrete sampler here wraps one such class;
    Mutation depends only on this interface.
    """

    @abstractmethod
    def sample(self, data: Data, column: str) -> Any:

        """
        Draws a plausible random value for `column` in `data`.

        Args
        ----
        - **data (Data)**: The dataset containing the column.
        - **column (str)**: The name of the column.

        Returns
        -------
        A randomly sampled value (a float within the column's observed
        range for a numerical sampler, a known category label for a
        categorical one).
        """

        pass


class NumericalValueSampler(ValueSampler):

    """Samples uniformly within [min, max] of a numerical column."""

    def sample(self, data: Data, column: str) -> Any:
        numerical_column = NumericalColumn(data, column)
        return random.uniform(numerical_column.min_val, numerical_column.max_val)


class CategoricalValueSampler(ValueSampler):

    """Samples uniformly from the known categories of a categorical column."""

    def sample(self, data: Data, column: str) -> Any:
        return random.choice(CategoricalColumn(data, column).classes)
