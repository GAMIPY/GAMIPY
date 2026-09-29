import random
from typing import Any

import numpy as np
import pandas as pd

from GAMIPY.data.data import Data
from GAMIPY.data.missing_index import MissingCell
from GAMIPY.population.gene import Gene


class Individual:

    """
    A base class for individuals in a genetic algorithm.

    Attributes
    ----------
    - **id (int)**: Unique identifier for the individual.
    - **score (float)**: The score of the individual, initialized to the maximum float value.
    - **genes (list)**: A list of Gene objects representing the individual's genes
    - **missing_indices (list)**: A list of tuples indicating the indices of missing values in the
      dataset, used for initializing genes.
    - **data (Data)**: The dataset from which the individual is created, used for
      random initialization of genes.


    Methods
    -------
    - **individual()**: Fills ``genes`` according to the subclass's initialization strategy (a no-op
      on a plain Individual, e.g. a crossover child).
    - **get_score()**: Returns the score of the individual.
    - **set_score()**: Sets the score of the individual to a specified value.
    - **__repr__()**: Returns a string representation of the Individual object, including its ID,
      genes, and score.
    """

    # Score of an individual not yet evaluated by Fitness. Kept as the max
    # float so min()/sorting treat it as the worst distance-based score, but
    # it must never be summed: callers that aggregate scores (RouletteWheel,
    # StochasticUniversal) filter on `is_evaluated` first, since summing two
    # of these overflows to inf.
    UNEVALUATED_SCORE: float = float(np.finfo(np.float64).max)

    def __init__(self, id: int, missing_indices: list[MissingCell], data: Data) -> None:

        """
        Initializes the Individual object with a unique ID and a list of missing indices.

        Args
        ----
        - **id (int)**: Unique identifier for the individual.
        - **missing_indices (list)**: A list of tuples indicating the indices of missing values in the dataset,
          where each tuple contains (row index, row index for missing value, column name).
        - **data (Data)**: The dataset from which the individual is created. It should contain the
          columns specified in the missing_indices list.

        Raises
        ------
        - **ValueError**: If the missing_indices list is empty or not provided.
        """

        if not missing_indices:
            raise ValueError("Missing indices must be provided for individual initialization.")
        self.id = id
        self.score = Individual.UNEVALUATED_SCORE
        self.genes: list[Gene] = []
        self.missing_indices = missing_indices
        self.data = data
        # Where to sample a column from when `data` (usually GAMIPY's m_data,
        # the rows with any missing value) has no observed value in it - e.g.
        # a dataset with a single missing cell. Set by the owning Population;
        # see _sampling_data().
        self.fallback_data: Data | None = None


    def gene_key(self) -> tuple[Any, ...]:
        """The individual's gene values, in gene order. Two individuals with the
        same key are the same imputation; the GA keeps keys unique within a
        population (see GeneticAlgorithmRunner)."""
        return tuple(gene.data for gene in self.genes)

    def _sampling_data(self, col: str) -> Data:
        """
        Returns the data to draw values for `col` from: `data` (usually
        m_data) whenever it has at least two distinct observed values in that
        column - unchanged behavior for realistic amounts of missing data -
        else `fallback_data` (the full original data). A column with no
        observed value in m_data (e.g. a single missing cell), or a single
        repeated one (e.g. two missing rows), would otherwise give every
        individual the same value, so the GA couldn't build distinct
        individuals.

        Raises
        ------
        - **ValueError**: If neither has an observed value in `col`.
        """
        profile = self.data.column_profile(col)
        if profile.distinct >= 2:
            return self.data
        if self.fallback_data is not None and self.fallback_data.column_profile(col).observed:
            return self.fallback_data
        if profile.observed:
            return self.data
        raise ValueError(f"Column '{col}' has no observed values to impute from.")

    def random_values(self, cells: list[MissingCell]) -> list[Any]:
        """
        A random value for each of `cells` (``(row, row_m, column)``, e.g.
        ``missing_indices`` or a subset of it): uniform within the column's
        observed range for a numerical column, one of its observed labels for
        a categorical one - the real label, not its code, so genes can be
        written straight back into the user's DataFrame. Values are drawn
        from `data`, or `fallback_data` (see `_sampling_data`).
        """
        # Column types and profiles are looked up once per column: asking
        # Data for them per gene used to take a third of a GA run.
        numeric = set(self.data.get_numeric_columns)
        profiles = {col: self._sampling_data(col).column_profile(col) for col in {cell[2] for cell in cells}}
        bounds = [(profiles[col].minimum, profiles[col].maximum) for _, _, col in cells if col in numeric]
        # One vectorized draw: NumPy's legacy generator gives the same numbers,
        # in the same order, as one np.random.uniform call per numerical cell.
        drawn = iter(np.random.uniform(*zip(*bounds, strict=True)).tolist() if bounds else [])
        return [next(drawn) if col in numeric else random.choice(profiles[col].classes) for _, _, col in cells]

    def individual(self) -> None:
        """
        Fills `genes` according to the subclass's initialization strategy.

        A plain Individual is a valid, concrete object whose genes are
        assigned directly (e.g. crossover children), so this base version
        is a deliberate no-op rather than an @abstractmethod - Individual is
        not an ABC, and the decorator only misled type checkers.
        """
    @property
    def get_score(self) -> float:
        """
        Returns the score of the individual.

        Returns
        -------
        - **float**: The score of the individual, which is used for evaluation in genetic algorithms.
        """
        return self.score


    def set_score(self, score:float) -> None:
        """
        Sets the score of the individual to a specified value.

        Args
        ----
        - **score (float)**: The score to be set for the individual.

        """
        self.score = score

    @property
    def is_evaluated(self) -> bool:
        """
        Whether the individual has been scored by Fitness.

        Returns
        -------
        - **bool**: False while the score is still `UNEVALUATED_SCORE`.
        """
        return self.score != Individual.UNEVALUATED_SCORE

    def __repr__(self) -> str:

        """
        Returns a string representation of the Individual object, including its ID, genes, and score.

        Returns
        -------
        - **str**: A string representation of the Individual object.
        """

        return f"Individual(id={self.id}, individual={self.genes}, score={self.score})"

class RandomIndividual(Individual):

    """
    Implements a random initialization strategy for individuals in a genetic algorithm.

    Attributes
    ----------
    - **id (int)**: Unique identifier for the individual.
    - **missing_indices (list)**: A list of tuples indicating the indices of missing values in the dataset.
    - **data (Data)**: The dataset from which the individual is created.

    Methods
    -------
    - **individual()**: Returns the individual's genes or representation.
    - **get_score()**: Returns the score of the individual.
    - **set_score()**: Sets the score of the individual to a specified value.
    - **__repr__()**: Returns a string representation of the Individual object, including its ID,
      genes, and score.
    """


    def individual(self) -> None:

        """
        Initializes the individual's genes with random values based on the missing indices
        and the data type of the columns in the dataset.
        """

        numeric = set(self.data.get_numeric_columns)
        values = self.random_values(self.missing_indices)
        for (row, row_m, col), value in zip(self.missing_indices, values, strict=True):
            dtype = "num" if col in numeric else "cat"
            self.genes.append(
                Gene(orig_location=(row, col), m_location=(row_m, col), name=col, data=value, dtype=dtype)
            )

class HeuristicIndividual(Individual):

    """
    Implements a heuristic initialization strategy for individuals in a genetic algorithm.

    Attributes
    ----------
    - **id (int)**: Unique identifier for the individual.
    - **missing_indices (list)**: A list of tuples indicating the indices of missing values in the dataset.
    - **data (Data)**: The dataset from which the individual is created.
    - **heuristic_population (pd.DataFrame)**: A DataFrame containing heuristic values for
      initializing the individual's genes.

    Methods
    -------
    - **individual()**: Returns the individual's genes or representation.
    - **get_score()**: Returns the score of the individual.
    - **set_score()**: Sets the score of the individual to a specified value.
    - **__repr__()**: Returns a string representation of the Individual object, including its ID,
      genes, and score.
    """

    def __init__(
        self, id: int, missing_indices: list[MissingCell], data: Data, heuristic_population: pd.DataFrame
    ) -> None:
        """
        Initializes a heuristic individual object with the specified parameters.

        Args
        ----
        - **id (int)**: Unique identifier for the individual.
        - **missing_indices (list)**: A list of tuples indicating the indices of missing values in the dataset.
        - **data (Data)**: The dataset from which the individual is created.
        - **heuristic_population (pd.DataFrame)**: A DataFrame containing heuristic values for
          initializing the individual's genes.
        """
        super().__init__(id, missing_indices, data)
        self.heuristic_population = heuristic_population

    def individual(self) -> None:
        """
        Initializes the individual's genes using heuristic values from a provided DataFrame.

        Raises
        ------
        - **ValueError**: If the heuristic_population is None or empty, or if it does not match the
          expected number of genes.
        - **ValueError**: If the heuristic_population is not a DataFrame or does not contain the expected columns.
        """

        if self.heuristic_population is None or self.heuristic_population.empty:
            raise ValueError("Heuristic population must be provided for heuristic initialization.")
        # Positional (row, row_m, column, value) tuples; iterrows built a
        # pandas Series per missing cell, every time a population was seeded.
        for row_index, m_row_index, col_name, value in self.heuristic_population.itertuples(index=False, name=None):
            # np.number too: a seed read from an integer column holds NumPy
            # integers, which aren't Python ints and used to count as "cat".
            dtype = "num" if isinstance(value, (int, float, np.number)) else "cat"

            gene = Gene(
                orig_location=(row_index, col_name),
                m_location=(m_row_index, col_name),
                name=col_name,
                data=value,
                dtype=dtype,
            )

            self.genes.append(gene)


class ConstantIndividual(Individual):
    """
    Implements a constant initialization strategy for individuals in a genetic algorithm.

    Attributes
    ----------
    - **id (int)**: Unique identifier for the individual.
    - **missing_indices (list)**: A list of tuples indicating the indices of missing values in the dataset.
    - **data (Data)**: The dataset from which the individual is created.
    - **constant_value (float)**: The constant value to be used for initializing the individual's genes.

    Methods
    -------
    - **individual()**: Returns the individual's genes or representation.
    - **get_score()**: Returns the score of the individual.
    - **set_score()**: Sets the score of the individual to a specified value.
    - **__repr__()**: Returns a string representation of the Individual object, including its ID,
      genes, and score.
    """

    def __init__(self, id: int, missing_indices: list[MissingCell], data: Data, constant_value: float) -> None:
        """
        Initializes a constant individual object with the specified parameters.

        Args:
        - **id (int)**: Unique identifier for the individual.
        - **missing_indices (list)**: A list of tuples indicating the indices of missing values in the dataset.
        - **data (Data)**: The dataset from which the individual is created.
        - **constant_value (float)**: The constant value to be used for initializing the individual's genes.

        """

        super().__init__(id, missing_indices, data)
        self.constant_value = constant_value

    def individual(self) -> None:
        """
        Initializes the individual's genes with a constant value for the specified missing indices.

        Raises
        ------
        - **ValueError**: If the constant_value is None or missing_indices is None.
        """

        if self.constant_value is None:
            raise ValueError("Constant value must be provided for constant initialization.")
        for row, row_m, col in self.missing_indices:
            if col in self.data.get_numeric_columns:
                gene = Gene(
                    orig_location=(row, col), m_location=(row_m, col), name=col, data=self.constant_value, dtype="num"
                )
            else:
                # The constant value (0.0/1.0/etc.) has no meaning as a category
                # label, so fall back to the column's most frequent category - a
                # real, in-domain value - rather than writing a numeric constant
                # that CategoricalCodec.encode's .map() would silently turn into
                # NaN because it isn't one of the column's known classes.
                mode = self._sampling_data(col).column_profile(col).mode
                gene = Gene(orig_location=(row, col), m_location=(row_m, col), name=col, data=mode, dtype="cat")
            self.genes.append(gene)


class ZerosIndividual(ConstantIndividual):
    """
    Implements a zeros initialization strategy for individuals in a genetic algorithm.

    Attributes
    ----------
    - **id (int)**: Unique identifier for the individual.
    - **missing_indices (list)**: A list of tuples indicating the indices of missing values in the dataset.
    - **data (Data)**: The dataset from which the individual is created.
    - **constant_value (float)**: The constant value to be used for initializing the individual's genes.

    Methods
    -------
    - **individual()**: Returns the individual's genes or representation.
    - **get_score()**: Returns the score of the individual.
    - **set_score()**: Sets the score of the individual to a specified value.
    - **__repr__()**: Returns a string representation of the Individual object, including its ID,
      genes, and score.
    """

    def __init__(self, id: int, missing_indices: list[MissingCell], data: Data) -> None:
        """
        Initializes a zeros individual object with the specified parameters.

        Args:
        - **id (int)**: Unique identifier for the individual.
        - **missing_indices (list)**: A list of tuples indicating the indices of missing values in the dataset.
        - **data (Data)**: The dataset from which the individual is created.
        """
        super().__init__(id, missing_indices, data, 0.0)


class OnesIndividual(ConstantIndividual):
    """
    Implements a ones initialization strategy for individuals in a genetic algorithm.

    Attributes
    ----------
    - **id (int)**: Unique identifier for the individual.
    - **missing_indices (list)**: A list of tuples indicating the indices of missing values in the dataset.
    - **data (Data)**: The dataset from which the individual is created.
    - **constant_value (float)**: The constant value to be used for initializing the individual's genes.

    Methods
    -------
    - **individual()**: Returns the individual's genes or representation.
    - **get_score()**: Returns the score of the individual.
    - **set_score()**: Sets the score of the individual to a specified value.
    - **__repr__()**: Returns a string representation of the Individual object, including its ID,
      genes, and score.
    """

    def __init__(self, id: int, missing_indices: list[MissingCell], data: Data) -> None:
        """
        Initializes a ones individual object with the specified parameters.

        Args:
        - **id (int)**: Unique identifier for the individual.
        - **missing_indices (list)**: A list of tuples indicating the indices of missing values in the dataset.
        - **data (Data)**: The dataset from which the individual is created.
        """
        super().__init__(id, missing_indices, data, 1.0)
