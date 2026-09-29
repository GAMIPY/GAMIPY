
import random
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from GAMIPY.data.data import Data
from GAMIPY.data.missing_index import MissingCell, MissingIndexLocator
from GAMIPY.population.gene import Gene
from GAMIPY.population.individual import (
    ConstantIndividual,
    HeuristicIndividual,
    Individual,
    OnesIndividual,
    RandomIndividual,
    ZerosIndividual,
)


class Population(ABC):

    """
    A base class for representing a population of individuals in a genetic algorithm.

    Attributes
    ----------
    - **data (Data)**: The dataset used for initializing the population.
    - **population (list)**: A list of Individual objects representing the population.
    - **population_size (int)**: The size of the population, default is 100.

    Methods
    -------
    - **initialize()**: Initializes the population based on the specified initialization type.
    - **create**: Builds the strategy registered under a name, passing each one only the extra options it declares.

    """

    # Names of the extra constructor arguments a subclass needs beyond
    # (population_size, data, missing_indices), e.g. ConstantPopulation's
    # constant. Population.create passes each strategy only the options it
    # declares, like Selection.create does.
    options: tuple[str, ...] = ()

    @staticmethod
    def create(
        strategy: str,
        population_size: int,
        data: Data,
        missing_indices: list[MissingCell] | None = None,
        fallback_data: Data | None = None,
        **options: Any,
    ) -> "Population":
        """
        Builds the initialization strategy registered under `strategy`, so
        callers depend on this abstraction rather than on the concrete
        subclasses.

        Args
        ----
        - **strategy (str)**: A key of POPULATION_STRATEGIES, e.g. 'random' or 'heuristic'.
        - **population_size (int)**: The number of individuals to build.
        - **data (Data)**: The dataset used for initializing the population.
        - **missing_indices (list)**: Missing-cell locations, see `__init__`.
        - **fallback_data (Data)**: The full original data, sampled for a column only when `data`
          has no observed value in it (see Individual._sampling_data).
        - **options**: Strategy-specific settings (e.g. heuristic_population, constant). Only the
          ones the chosen strategy declares in its `options` are passed on; the rest are ignored.

        Raises
        ------
        - **ValueError**: If `strategy` is not a registered initialization strategy.
        """

        try:
            cls = POPULATION_STRATEGIES[strategy]
        except KeyError:
            raise ValueError(
                f"Invalid initialization strategy. Choose from {', '.join(sorted(POPULATION_STRATEGIES))}."
            ) from None
        own_options = {name: options[name] for name in cls.options if name in options}
        population = cls(population_size=population_size, data=data, missing_indices=missing_indices, **own_options)
        population.fallback_data = fallback_data
        return population

    def __init__(self, population_size: int, data: Data, missing_indices: list[MissingCell] | None = None) -> None:
        """
        Initializes the Population object with specified values.

        """
        self.data = data
        self.population_size = population_size
        self.population: list[Individual] = []
        # Handed to every individual built here; see
        # Individual._sampling_data(). Set via Population.create(fallback_data=...).
        self.fallback_data: Data | None = None
        # `missing_indices` should be computed from the caller's true original
        # Data (see GAMIPY.__init__), not from `data` here (which GAMIPY passes
        # as m_data, an already row-filtered, reset-index subset) - otherwise
        # Gene.orig_location ends up m_data-relative instead of a real row in
        # the user's original DataFrame. Falls back to the old self-computed
        # behavior when not provided, for callers that construct Population
        # subclasses directly with their own (already-original) Data.
        self.missing_indices = missing_indices if missing_indices is not None else MissingIndexLocator(data).locate()

    @abstractmethod
    def initialize(self) -> list[Individual]:
        """
        Initializes the population based on the specified initialization type.
        """
        pass

    def clear_population(self) -> None:
        """
        Clears the current population by resetting the population list to an empty list.
        """

        self.population = []

    def add_individual(self, individual: Individual) -> None:
        """
        Adds an individual to the population.

        Args
        ----
        - **individual (Individual)**: The individual to be added to the population.

        Raises
        ------
        - **ValueError**: If the individual is not an instance of the Individual class.
        """

        if not isinstance(individual, Individual):
            raise ValueError("Only Individual instances can be added to the population.")
        self.population.append(individual)


class RandomPopulation(Population):
    """
    Implements a random initialization strategy for the population.

    Attributes
    ----------
    - **id (int)**: Unique identifier for the individual.
    - **missing_indices (list)**: A list of tuples indicating the indices of missing values in the dataset.
    - **data (Data)**: The dataset from which the individual is created.

    Methods
    -------
    - **initialize()**: Initializes the population with random values.
    """

    def initialize(self) -> list[Individual]:
        """
        Initializes the population with random values based on the missing indices
        and the data type of the columns in the dataset.

        Returns
        -------
        - **list**: A list of Individual objects representing the initialized population.
        """

        missing_indices = self.missing_indices

        for i in range(self.population_size):
            temp_ind = RandomIndividual(id=i, missing_indices=missing_indices, data=self.data)
            temp_ind.fallback_data = self.fallback_data
            temp_ind.individual()

            self.population.append(temp_ind)

        return self.population


class HeuristicPopulation(Population):
    """
    Implements a heuristic initialization strategy for the population.

    Attributes
    ----------
    - **id (int)**: Unique identifier for the individual.
    - **missing_indices (list)**: A list of tuples indicating the indices of missing values in the dataset.
    - **data (Data)**: The dataset from which the individual is created.
    - **heuristic_population (DataFrame)**: A DataFrame containing heuristic values for initialization.

    Methods
    -------
    - **initialize()**: Initializes the population with heuristic values.
    """

    options = ("heuristic_population",)

    def __init__(
        self,
        population_size: int,
        data: Data,
        heuristic_population: pd.DataFrame,
        missing_indices: list[MissingCell] | None = None,
    ):
        """
        Initializes the HeuristicIndividual object with the provided parameters.

        Args
        ----
        - **id (int)**: Unique identifier for the individual.
        - **missing_indices (list)**: A list of tuples indicating the indices of missing values in the dataset.
        - **data (Data)**: The dataset from which the individual is created.
        - **heuristic_population (DataFrame)**: A DataFrame containing heuristic values for initialization.
        """

        super().__init__(population_size, data, missing_indices)
        self.heuristic_population = heuristic_population

    def initialize(self) -> list[Individual]:
        """
        Initializes the population using heuristic values from a provided DataFrame.

        Returns
        -------
        - **list**: A list of Individual objects representing the initialized population.

        Raises
        ------
        - **ValueError**: If the heuristic population is None or empty.
        """

        if self.heuristic_population is None or self.heuristic_population.empty:
            raise ValueError("Heuristic population must be provided for heuristic initialization.")
        if self.population_size < 1:
            return self.population
        # One copy of the seed plus distinct variants of it: identical copies
        # add nothing to the search (crossing two of them gives the same
        # individual back), and the GA keeps every individual unique.
        seed = HeuristicIndividual(
            id=0, missing_indices=self.missing_indices, data=self.data, heuristic_population=self.heuristic_population
        )
        seed.fallback_data = self.fallback_data
        seed.individual()
        self.population.append(seed)
        for i in range(1, self.population_size):
            self.population.append(self._variant(seed, i))
        return self.population

    # Share of the seed's genes each variant resamples at random (at least one).
    variant_rate = 0.2

    def _variant(self, seed: Individual, id: int) -> Individual:
        """A copy of `seed` with about `variant_rate` of its genes (and always
        at least one) replaced by a random value for that cell, drawn the same
        way RandomIndividual draws them. Only the replaced genes are drawn
        (this used to build a whole random individual per variant)."""

        n_genes = len(seed.genes)
        resampled = sorted(
            {i for i in range(n_genes) if random.random() < self.variant_rate} or {random.randrange(n_genes)}
        )
        variant = Individual(id, missing_indices=self.missing_indices, data=self.data)
        variant.fallback_data = self.fallback_data
        # The seed's own cells: its genes follow heuristic_population's row
        # order, which need not be missing_indices' order. A seed gene for a
        # cell that isn't missing in `data` keeps its value.
        missing = {(row_m, col) for _, row_m, col in self.missing_indices}
        drawn = [i for i in resampled if seed.genes[i]._m_location in missing]
        cells = [(seed.genes[i].location[0], seed.genes[i]._m_location[0], seed.genes[i].name) for i in drawn]
        new_values = dict(zip(drawn, variant.random_values(cells), strict=True))
        variant.genes = [
            Gene(
                orig_location=gene.location,
                m_location=gene._m_location,
                name=gene.name,
                data=new_values.get(i, gene.data),
                dtype=gene.dtype,
            )
            for i, gene in enumerate(seed.genes)
        ]
        return variant


class ZerosPopulation(Population):
    """
    Implements a zero initialization strategy for the population.

    Attributes
    ----------
    - **id (int)**: Unique identifier for the individual.
    - **missing_indices (list)**: A list of tuples indicating the indices of missing values in the dataset.
    - **data (Data)**: The dataset from which the individual is created.

    Methods
    -------
    - **initialize()**: Initializes the population with zeros.
    """

    def initialize(self) -> list[Individual]:
        """
        Initializes the population with zeros for the specified missing indices.

        Returns
        -------
        - **list**: A list of Individual objects representing the initialized population.
        """

        for i in range(self.population_size):
            temp_ind = ZerosIndividual(id=i, missing_indices=self.missing_indices, data=self.data)
            temp_ind.fallback_data = self.fallback_data
            temp_ind.individual()
            self.population.append(temp_ind)

        return self.population


class OnesPopulation(Population):
    """
    Implements a ones initialization strategy for the population.

    Attributes
    ----------
    - **id (int)**: Unique identifier for the individual.
    - **missing_indices (list)**: A list of tuples indicating the indices of missing values in the dataset.
    - **data (Data)**: The dataset from which the individual is created.

    Methods
    -------
    - **initialize()**: Initializes the population with ones.
    """

    def initialize(self) -> list[Individual]:
        """
        Initializes the population with ones for the specified missing indices.

        Returns
        -------
        - **list**: A list of Individual objects representing the initialized population.
        """

        for i in range(self.population_size):
            temp_ind = OnesIndividual(id=i, missing_indices=self.missing_indices, data=self.data)
            temp_ind.fallback_data = self.fallback_data
            temp_ind.individual()
            self.population.append(temp_ind)

        return self.population


class ConstantPopulation(Population):
    """
    Implements a constant initialization strategy for the population.

    Attributes
    ----------
    - **id (int)**: Unique identifier for the individual.
    - **missing_indices (list)**: A list of tuples indicating the indices of missing values in the dataset.
    - **data (Data)**: The dataset from which the individual is created.
    - **constant (float)**: The constant value to use for initialization.

    Methods
    -------
    - **initialize()**: Initializes the population with a constant value.
    """

    options = ("constant",)

    def __init__(
        self, population_size: int, data: Data, constant: float, missing_indices: list[MissingCell] | None = None
    ) -> None:
        """
        Initializes the population with a constant value for the specified missing indices.

        Args
        ----
        - **id (int)**: Unique identifier for the individual.
        - **missing_indices (list)**: A list of tuples indicating the indices of missing values in the dataset.
        - **data (Data)**: The dataset from which the individual is created.
        - **constant (float)**: The constant value to use for initialization.
        """

        super().__init__(population_size, data, missing_indices)
        self.constant = constant

    def initialize(self) -> list[Individual]:
        """
        Initializes the population with a constant value for the specified missing indices.

        Returns
        -------
        - **list**: A list of Individual objects representing the initialized population.

        Raises
        ------
        - **ValueError**: If the constant value is None.
        """

        if self.constant is None:
            raise ValueError("Constant value must be provided for constant initialization.")

        for i in range(self.population_size):
            temp_ind = ConstantIndividual(
                id=i, missing_indices=self.missing_indices, data=self.data, constant_value=self.constant
            )
            temp_ind.fallback_data = self.fallback_data
            temp_ind.individual()
            self.population.append(temp_ind)

        return self.population


class HybridPopulation(Population):
    """
    Implements a hybrid initialization strategy for the population.

    Attributes
    ----------
    - **id (int)**: Unique identifier for the individual.
    - **missing_indices (list)**: A list of tuples indicating the indices of missing values in the dataset.
    - **data (Data)**: The dataset from which the individual is created.
    - **heuristic_population (np.ndarray)**: The heuristic population to use for initialization.

    Methods
    -------
    - **initialize()**: Initializes the population using a combination of random and heuristic initialization methods.
    """

    options = ("heuristic_population",)

    def __init__(
        self,
        population_size: int,
        data: Data,
        heuristic_population: pd.DataFrame | None = None,
        missing_indices: list[MissingCell] | None = None,
    ) -> None:
        """
        Initializes the hybrid population with the specified size, data, and heuristic population.

        Args
        ----
        - **population_size (int)**: The size of the population.
        - **data (Data)**: The dataset to use for initialization.
        - **heuristic_population (np.ndarray)**: The heuristic population to use for initialization.
        """

        super().__init__(population_size, data, missing_indices)
        self.heuristic_population = heuristic_population

    def initialize(self) -> list[Individual]:
        """
        Initializes the population using a combination of random and heuristic initialization
        methods. It creates a population that consists of a specified number of individuals
        initialized randomly and the rest initialized using heuristic values.

        Returns
        -------
        - **list**: A list of Individual objects representing the initialized population.

        Raises
        ------
        - **ValueError**: If the heuristic population is None or empty, or if the hybrid population
          size does not match the specified population size.
        """

        # Half of the individuals (rounded up, so there is always a seed) come
        # from the heuristic seed - the seed itself plus variants of it, see
        # HeuristicPopulation - and the rest are random. This used to take
        # heuristic_population's row count (= the number of missing cells) as
        # the number of seeded individuals, which made the split depend on
        # the dataset and broke when there were more cells than individuals.
        num_heuristic = (self.population_size + 1) // 2
        num_random = self.population_size - num_heuristic

        if self.heuristic_population is None or self.heuristic_population.empty:
            raise ValueError("Heuristic population must be provided for hybrid initialization.")
        hp = HeuristicPopulation(
            population_size=num_heuristic,
            data=self.data,
            heuristic_population=self.heuristic_population,
            missing_indices=self.missing_indices,
        )
        hp.fallback_data = self.fallback_data
        heuristic_population = hp.initialize()

        random_population: list[Individual] = []
        for i in range(num_random):
            temp_ind = RandomIndividual(id=num_heuristic + i, missing_indices=self.missing_indices, data=self.data)
            temp_ind.fallback_data = self.fallback_data
            temp_ind.individual()
            random_population.append(temp_ind)

        self.population = heuristic_population + random_population

        if len(self.population) != self.population_size:
            raise ValueError("Hybrid population size does not match the specified population size.")

        return self.population


# Strategy name -> population, used by Population.create. Registering a new
# strategy here (with its own `options`) is all GAMIPY needs to support it.
POPULATION_STRATEGIES: dict[str, type[Population]] = {
    "random": RandomPopulation,
    "heuristic": HeuristicPopulation,
    "zeros": ZerosPopulation,
    "ones": OnesPopulation,
    "constant": ConstantPopulation,
    "hybrid": HybridPopulation,
}
