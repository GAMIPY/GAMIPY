import random
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from GAMIPY.population.gene import Gene
from GAMIPY.population.individual import Individual


class Crossover(ABC):
    """
    A base class for crossover operations in genetic algorithms.

    Attributes
    ----------

        - **crossover_rate (float)**: The probability of crossover occurring between two individuals.
        - **population (list)**: A list of individuals in the population, used to check if all genes
          are numerical for arithmetic and blend crossover.

    Methods
    -------
        - **run(parent1, parent2)**: Abstract method to be implemented by subclasses for performing crossover.
        - **create**: Builds the crossover registered under a type name.

    """

    @staticmethod
    def create(crossover_type: str, crossover_rate: float, population: list[Individual]) -> "Crossover":

        """
        Builds the crossover registered under `crossover_type`, mirroring
        Selection.create, so callers depend on this abstraction rather than
        on the concrete subclasses.

        Args
        ----
            - **crossover_type (str)**: A key of CROSSOVER_TYPES, e.g. 'uniform' or 'blend'.
            - **crossover_rate (float)**: The probability of crossover occurring between two individuals.
            - **population (list)**: A list of individuals (not a Population object), see `__init__`.

        Raises
        ------
            - **ValueError**: If `crossover_type` is not a registered crossover type.
        """

        try:
            strategy = CROSSOVER_TYPES[crossover_type]
        except KeyError:
            raise ValueError(
                "Invalid crossover type. Choose from "
                f"{', '.join(sorted(CROSSOVER_TYPES))}."
            ) from None
        return strategy(crossover_rate=crossover_rate, population=population)

    def __init__(self, crossover_rate: float = 0.2, population: list[Individual] | None = None) -> None:

        """
        Initializes the Crossover object with a crossover rate and population.

        Args
        ----
            - **crossover_rate (float)**: The probability of crossover occurring between two
              individuals. Default is 0.2 (20% chance of crossover).
            - **population (list)**: A list of individuals in the population, used to check if all
              genes are numerical for arithmetic and blend crossover. Default is an empty list.

        """


        self.crossover_rate = crossover_rate
        self.population = population if population is not None else []

    @abstractmethod
    def run(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        """
        Performs the crossover operation between two parent individuals and returns two child individuals.

        Args
        ----
            - **parent1 (Individual)**: The first parent individual.
            - **parent2 (Individual)**: The second parent individual.

        Returns
        -------
            - **tuple[Individual, Individual]**: A tuple containing two child individuals resulting
              from the crossover operation.

        """
        pass


class Uniform(Crossover):

    """
    Implements uniform crossover between two parent individuals.

    Attributes
    ----------
        - **crossover_rate (float)**: The probability of crossover occurring between two individuals.
        - **population (list)**: A list of individuals in the population, used to check if all genes
          are numerical for arithmetic and blend crossover.

    Methods
    -------
        - **run(parent1, parent2)**: Performs uniform crossover between two parent individuals.

    """


    def run(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:

        """
        Performs uniform crossover between two parent individuals.

        Args
        ----
            - **parent1 (Individual)**: The first parent individual.
            - **parent2 (Individual)**: The second parent individual.


        Returns
        -------
            - **tuple[Individual, Individual]**: A tuple containing two child individuals resulting
              from the uniform crossover operation.

        """
        child1 = Individual(parent1.id,missing_indices=parent1.missing_indices,data=parent1.data)
        child2 = Individual(parent2.id,missing_indices=parent2.missing_indices,data=parent2.data)
        for gene1, gene2 in zip(parent1.genes, parent2.genes, strict=True):
            random_choice = random.random()

            if random_choice < self.crossover_rate:

                child1.genes.append(
                    Gene(
                        orig_location=gene2.location,
                        m_location=gene2._m_location,
                        name=gene2.name,
                        data=gene2.data,
                        dtype=gene2.dtype,
                    )
                )
                child2.genes.append(
                    Gene(
                        orig_location=gene1.location,
                        m_location=gene1._m_location,
                        name=gene1.name,
                        data=gene1.data,
                        dtype=gene1.dtype,
                    )
                )
            else:
                child1.genes.append(
                    Gene(
                        orig_location=gene1.location,
                        m_location=gene1._m_location,
                        name=gene1.name,
                        data=gene1.data,
                        dtype=gene1.dtype,
                    )
                )
                child2.genes.append(
                    Gene(
                        orig_location=gene2.location,
                        m_location=gene2._m_location,
                        name=gene2.name,
                        data=gene2.data,
                        dtype=gene2.dtype,
                    )
                )

        child1.set_score(Individual.UNEVALUATED_SCORE)
        child2.set_score(Individual.UNEVALUATED_SCORE)
        return child1, child2


class OnePoint(Crossover):
    """
    Implements one-point crossover between two parent individuals.

    Attributes
    ----------
        - **crossover_rate (float)**: The probability of crossover occurring between two individuals.
        - **population (list)**: A list of individuals in the population, used to check if all genes
          are numerical for arithmetic and blend crossover.

    Methods
    -------
        - **run(parent1, parent2)**: Performs uniform crossover between two parent individuals.

    """

    def run(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        """
        Performs one-point crossover between two parent individuals.

        Args
        ----
            - **parent1 (Individual)**: The first parent individual.
            - **parent2 (Individual)**: The second parent individual.

        Returns
        -------
            - **tuple[Individual, Individual]**: A tuple containing two child individuals resulting
              from the one-point crossover operation.

        """
        # A single gene has no interior cut point (randint(1, 0) raises), so
        # the children are plain copies of their parents.
        n_genes = len(parent1.genes)
        point = random.randint(1, n_genes - 1) if n_genes >= 2 else n_genes
        child1 = Individual(parent1.id, missing_indices=parent1.missing_indices, data=parent1.data)
        child2 = Individual(parent2.id, missing_indices=parent2.missing_indices, data=parent2.data)
        child1.genes = parent1.genes[:point] + parent2.genes[point:]
        child2.genes = parent2.genes[:point] + parent1.genes[point:]

        child1.set_score(Individual.UNEVALUATED_SCORE)
        child2.set_score(Individual.UNEVALUATED_SCORE)
        return child1, child2


class TwoPoint(Crossover):
    """
    Implements two-point crossover between two parent individuals.

    Attributes
    ----------
        - **crossover_rate (float)**: The probability of crossover occurring between two individuals.
        - **population (list)**: A list of individuals in the population, used to check if all genes
          are numerical for arithmetic and blend crossover.

    Methods
    -------
        - **run(parent1, parent2)**: Performs uniform crossover between two parent individuals.


    """

    def run(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        """
        Performs two-point crossover between two parent individuals.

        Args
        ----
            - **parent1 (Individual)**: The first parent individual.
            - **parent2 (Individual)**: The second parent individual.

        Returns
        -------
            - **tuple[Individual, Individual]**: A tuple containing two child individuals resulting
              from the two-point crossover operation.

        """

        # Two interior cut points need at least 3 genes (randint(1, 0)
        # raises otherwise), which a dataset with only 1-2 missing cells
        # doesn't have - fall back to one-point crossover there.
        if len(parent1.genes) < 3:
            return OnePoint(crossover_rate=self.crossover_rate, population=self.population).run(parent1, parent2)
        point1 = random.randint(1, len(parent1.genes) - 2)
        point2 = random.randint(point1 + 1, len(parent1.genes) - 1)
        child1 = Individual(parent1.id, missing_indices=parent1.missing_indices, data=parent1.data)
        child2 = Individual(parent2.id, missing_indices=parent2.missing_indices, data=parent2.data)
        child1.genes = parent1.genes[:point1] + parent2.genes[point1:point2] + parent1.genes[point2:]
        child2.genes = parent2.genes[:point1] + parent1.genes[point1:point2] + parent2.genes[point2:]

        child1.set_score(Individual.UNEVALUATED_SCORE)
        child2.set_score(Individual.UNEVALUATED_SCORE)
        return child1, child2


class NumericCrossover(Crossover):
    """
    Base for crossovers that combine gene values arithmetically (averaging,
    blending) and so only make sense for numerical genes. Split out of
    Crossover, where `is_all_numerical` was defined for every subclass
    although only Arithmetic and Blend use it.
    """

    def is_all_numerical(self, *individuals: Individual) -> bool:
        """
        Checks that every gene of the given individuals - or, if none are
        given, of every individual in `population` - is numerical
        (dtype "num").

        Returns
        -------
            - **bool**: True if all genes are numerical, False otherwise.
        """

        for ind in individuals or self.population:
            for gene in ind.genes:
                if gene.dtype != "num":
                    return False
        return True


class Arithmetic(NumericCrossover):
    """
    Implements arithmetic crossover between two parent individuals.

    Attributes
    ----------
        - **crossover_rate (float)**: The probability of crossover occurring between two individuals.
        - **population (list)**: A list of individuals in the population, used to check if all genes
          are numerical for arithmetic and blend crossover.

    Methods
    -------
        - **run(parent1, parent2)**: Performs uniform crossover between two parent individuals.

    """

    def run(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        """
        Performs arithmetic crossover between two parent individuals.

        Args
        ----
            - **parent1 (Individual)**: The first parent individual.
            - **parent2 (Individual)**: The second parent individual.

        Returns
        -------
            - **tuple[Individual, Individual]**: A tuple containing two child individuals resulting
              from the arithmetic crossover operation.
        """
        if not self.is_all_numerical(parent1, parent2):
            raise ValueError("Arithmetic crossover is only applicable to numerical genes.")

        child1 = Individual(parent1.id, missing_indices=parent1.missing_indices, data=parent1.data)
        child2 = Individual(parent2.id, missing_indices=parent2.missing_indices, data=parent2.data)

        for gene1, gene2 in zip(parent1.genes, parent2.genes, strict=True):
            new_data1 = (gene1.data + gene2.data) / 2
            new_data2 = (gene1.data + gene2.data) / 2
            child1.genes.append(
                Gene(
                    orig_location=gene1.location,
                    m_location=gene1._m_location,
                    name=gene1.name,
                    data=new_data1,
                    dtype=gene1.dtype,
                )
            )
            child2.genes.append(
                Gene(
                    orig_location=gene2.location,
                    m_location=gene2._m_location,
                    name=gene2.name,
                    data=new_data2,
                    dtype=gene2.dtype,
                )
            )

        child1.set_score(Individual.UNEVALUATED_SCORE)
        child2.set_score(Individual.UNEVALUATED_SCORE)
        return child1, child2


class Blend(NumericCrossover):
    """
    Implements blend crossover between two parent individuals.

    Attributes
    ----------
        - **crossover_rate (float)**: The probability of crossover occurring between two individuals.
        - **population (list)**: A list of individuals in the population, used to check if all genes
          are numerical for arithmetic and blend crossover.

    Methods
    -------
        - **run(parent1, parent2)**: Performs uniform crossover between two parent individuals.

    """

    def run(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        """
        Performs blend crossover between two parent individuals.

        Args
        ----
            - **parent1 (Individual)**: The first parent individual.
            - **parent2 (Individual)**: The second parent individual.

        Returns
        -------
            - **tuple[Individual, Individual]**: A tuple containing two child individuals resulting
              from the blend crossover operation.

        """
        if not self.is_all_numerical(parent1, parent2):
            raise ValueError("Blend crossover is only applicable to numerical genes.")

        alpha = random.uniform(0.5, 1.5)
        child1 = Individual(parent1.id, missing_indices=parent1.missing_indices, data=parent1.data)
        child2 = Individual(parent2.id, missing_indices=parent2.missing_indices, data=parent2.data)

        for gene1, gene2 in zip(parent1.genes, parent2.genes, strict=True):
            new_data1 = alpha * gene1.data + (1 - alpha) * gene2.data
            new_data2 = alpha * gene2.data + (1 - alpha) * gene1.data
            child1.genes.append(
                Gene(
                    orig_location=gene1.location,
                    m_location=gene1._m_location,
                    name=gene1.name,
                    data=new_data1,
                    dtype=gene1.dtype,
                )
            )
            child2.genes.append(
                Gene(
                    orig_location=gene2.location,
                    m_location=gene2._m_location,
                    name=gene2.name,
                    data=new_data2,
                    dtype=gene2.dtype,
                )
            )

        child1.set_score(Individual.UNEVALUATED_SCORE)
        child2.set_score(Individual.UNEVALUATED_SCORE)
        return child1, child2


# Type name -> crossover, used by Crossover.create. Registering a new
# crossover here is all GAMIPY needs to support it.
CROSSOVER_TYPES: dict[str, type[Crossover]] = {
    "uniform": Uniform,
    "one_point": OnePoint,
    "two_point": TwoPoint,
    "arithmetic": Arithmetic,
    "blend": Blend,
}


class ArithmeticModelCrossover:

    """
    Arithmetic crossover for the model-based GA: with probability
    ``crossover_rate`` the child's model for a target column is a random
    weighted average of the two parents' models, otherwise a copy of the first
    parent's model.

    Attributes
    ----------
    - **crossover_rate (float)**: Probability of crossing two parent models (default 0.1).
    """

    def __init__(self, crossover_rate: float = 0.1) -> None:
        self.crossover_rate = crossover_rate

    def cross(
        self,
        weights_a: npt.NDArray[np.float64],
        bias_a: npt.NDArray[np.float64],
        weights_b: npt.NDArray[np.float64],
        bias_b: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        The child's weights and bias for one target column: with probability
        ``crossover_rate``, ``share * a + (1 - share) * b`` with ``share``
        drawn uniformly from [0, 1); otherwise parent ``a``'s.

        Args
        ----
        - **weights_a, bias_a (np.ndarray)**: The first parent's model for the column.
        - **weights_b, bias_b (np.ndarray)**: The second parent's model for the column.
        """
        if np.random.random() < self.crossover_rate:
            share = np.random.random()
            return share * weights_a + (1 - share) * weights_b, share * bias_a + (1 - share) * bias_b
        return weights_a, bias_a
