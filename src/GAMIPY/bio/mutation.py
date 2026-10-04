import random

import numpy as np
import numpy.typing as npt

from GAMIPY.bio.value_sampling import CategoricalValueSampler, NumericalValueSampler, ValueSampler
from GAMIPY.data.data import Data
from GAMIPY.population.gene import Gene
from GAMIPY.population.individual import Individual
from GAMIPY.population.population import Population


class Mutation:

    """
    Mutation class for applying mutation operations on individuals in a genetic algorithm.
    This class allows for mutation of genes in an individual based on a specified mutation rate.

    The value drawn for a mutated gene is delegated to a `ValueSampler`
    (GAMIPY.bio.value_sampling), chosen by gene dtype, rather than
    `Mutation.mutate` instantiating `NumericalColumn`/`CategoricalColumn`
    directly.

    Attributes
    ----------
    - **mutation_rate (float)**: The probability of mutation occurring for each gene in an
      individual. Default is 0.1 (10% chance of mutation).
    - **population (list)**: A list of individuals in the population, used to apply mutation operations.
    - **data (Data)**: The dataset used for mutation operations, providing access to numerical and categorical columns.
    - **numeric_sampler (ValueSampler)**: Draws a mutated value for a numerical gene. Defaults to
      `NumericalValueSampler()`.
    - **categorical_sampler (ValueSampler)**: Draws a mutated value for a categorical gene. Defaults
      to `CategoricalValueSampler()`.

    Methods
    -------
    - **mutate(individual: Individual) -> Individual**: Returns a mutated copy of the individual (the
      input is left unchanged). Each gene is resampled with probability ``mutation_rate``: numerical
      genes within the column's observed range, categorical genes from its known labels.

    Raises
    ------
    - **ValueError**: If the gene type is unknown (not "num" or "cat"). This ensures that only
      numerical and categorical genes can be mutated.
    """

    def __init__(
        self,
        mutation_rate: float = 0.1,
        population: Population | None = None,
        data: Data | None = None,
        numeric_sampler: ValueSampler | None = None,
        categorical_sampler: ValueSampler | None = None,
        mutation_type: str = 'uniform',
        mutation_scale: float = 0.1,
    ) -> None:
        """
        Initializes the Mutation object with a mutation rate, population, and data.

        Args
        ----
        - **mutation_rate (float)**: The probability of mutation occurring for each gene in an
          individual. Default is 0.1 (10% chance of mutation).
        - **population (list)**: A list of individuals in the population, used to apply mutation
          operations. Default is an empty list.
        - **data (Data)**: The dataset used for mutation operations, providing access to numerical
          and categorical columns. Default is None, which means no data is provided.
        - **numeric_sampler (ValueSampler)**: Draws a mutated value for a numerical gene. Default is
          a new `NumericalValueSampler()`.
        - **categorical_sampler (ValueSampler)**: Draws a mutated value for a categorical gene.
          Default is a new `CategoricalValueSampler()`.
        - **mutation_type (str)**: ``'uniform'`` (default) draws a mutated numerical gene with
          ``numeric_sampler``; ``'gaussian'`` adds normal noise with standard deviation
          ``mutation_scale`` times the column's, clipped to the column's observed range.
        - **mutation_scale (float)**: Noise scale for ``'gaussian'`` (default 0.1).

        Raises
        ------
        - **ValueError**: If ``mutation_type`` is unknown.
        """

        self.mutation_rate = mutation_rate
        self.population = population
        self.data = data
        self.numeric_sampler = numeric_sampler if numeric_sampler is not None else NumericalValueSampler()
        self.categorical_sampler = categorical_sampler if categorical_sampler is not None else CategoricalValueSampler()
        if mutation_type not in ('uniform', 'gaussian'):
            raise ValueError(f"mutation_type must be 'uniform' or 'gaussian', got {mutation_type!r}.")
        self.mutation_type = mutation_type
        self.mutation_scale = mutation_scale
        self._column_ranges: dict[str, tuple[float, float, float]] = {}

    def _gaussian(self, value: float, column: str) -> float:
        """`value` plus normal noise scaled to the column, clipped to its observed range."""
        assert self.data is not None
        if column not in self._column_ranges:
            observed = self.data.get_column(column).dropna().astype(float)
            self._column_ranges[column] = (float(observed.min()), float(observed.max()), float(observed.std() or 0.0))
        low, high, std = self._column_ranges[column]
        return float(np.clip(value + np.random.normal(0.0, self.mutation_scale * std), low, high))

    def mutate(self, individual: Individual) -> Individual:
        """


        Applies mutation to the given individual. For each gene in the individual, if a random
        number is less than the mutation rate,
        the gene is mutated. Numerical genes are mutated to a random value within the
        range of the corresponding numerical column in the dataset, while categorical genes are
        mutated to a random value from the list of classes in the corresponding categorical column.

        Args
        ----
        - **individual (Individual)**: The individual whose genes will be mutated.

        Returns
        -------
        - **Individual**: A new, unevaluated individual with the mutated gene values. The input
          individual is left unchanged: it may be a survivor (or the best individual found so far)
          whose score must keep matching its genes.

        Raises
        ------
        - **ValueError**: If the gene type is unknown (not "num" or "cat"). This ensures that only
          numerical and categorical genes can be mutated.
        """
        if self.data is None:
            raise ValueError("Mutation.mutate() needs data to sample new values from.")
        mutant = Individual(individual.id, missing_indices=individual.missing_indices, data=individual.data)
        mutant.genes = [
            Gene(
                orig_location=gene.location,
                m_location=gene._m_location,
                name=gene.name,
                data=gene.data,
                dtype=gene.dtype,
            )
            for gene in individual.genes
        ]
        for gene in mutant.genes:
            random_choice = random.random()

            if random_choice < self.mutation_rate:
                if gene.dtype == "num" and self.mutation_type == "gaussian":
                    gene.data = self._gaussian(float(gene.data), gene.name)
                elif gene.dtype == "num":
                    gene.data = self.numeric_sampler.sample(self.data, gene.name)
                elif gene.dtype == "cat":
                    gene.data = self.categorical_sampler.sample(self.data, gene.name)
                else:
                    raise ValueError(f"Unknown gene type: {gene.dtype}")
        return mutant


class GaussianModelMutation:

    """
    Gaussian mutation for the model-based GA: each weight and bias of a child
    is perturbed, with probability ``mutation_rate``, by normal noise with
    standard deviation ``mutation_scale``. Weights that must stay zero (a
    model's own column) are reset to zero.

    Attributes
    ----------
    - **mutation_rate (float)**: Probability of mutating each weight and bias (default 0.01).
    - **mutation_scale (float)**: Standard deviation of the noise (default 0.1).
    """

    def __init__(self, mutation_rate: float = 0.01, mutation_scale: float = 0.1) -> None:
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale

    def mutate(
        self,
        weights: npt.NDArray[np.float64],
        bias: npt.NDArray[np.float64],
        allowed: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Mutated copies of a candidate's weights (outputs x features) and bias
        (outputs); `allowed` (same shape as `weights`) is 0 where a weight
        must stay zero and 1 elsewhere.
        """
        mutated = np.random.random(weights.shape) < self.mutation_rate
        noise = np.random.normal(0.0, self.mutation_scale, weights.shape)
        weights = np.where(mutated, weights + noise, weights) * allowed
        mutated_bias = np.random.random(bias.shape) < self.mutation_rate
        bias = np.where(mutated_bias, bias + np.random.normal(0.0, self.mutation_scale, bias.shape), bias)
        return weights, bias
