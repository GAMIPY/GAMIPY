import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pandas as pd

from .bio.crossover import Crossover
from .bio.distance import DistanceMetric
from .bio.selection import Selection
from .config import GAConfig
from .data.data import Data
from .datasets import sample_data
from .model_search import ModelSearch
from .parallel import thread_count
from .population.individual import Individual
from .population.population import Population
from .runner import GeneticAlgorithmRunner
from .strategy_factory import StrategyFactory


class GAMIPY:

    """
    Imputes the missing values of a DataFrame with a genetic algorithm.

    GAMIPY is a facade: the hyperparameters live in ``config`` (GAConfig),
    operators are built by StrategyFactory, the GA loop is run by
    GeneticAlgorithmRunner, and the result is written by ImputedDataBuilder.
    Change hyperparameters after construction through ``gamipy.config``.

    Parameters
    ----------
    - **data (pd.DataFrame)**: The data with missing values (NaN/None) to impute.
    - **population_size (int)**: Individuals per generation (default 100).
    - **generations (int)**: Number of generations (default 50).
    - **crossover_rate (float)**: Probability of crossover between two parents (default 0.1).
    - **mutation_rate (float)**: Probability of mutating each gene (default 0.01).
    - **selection_method (str)**: A key of ``SELECTION_METHODS`` (default ``'elitist'``).
    - **selection_number (int)**: Individuals selected per generation, at least 2 (default 10).
    - **fitness_method (str)**: A key of ``DISTANCE_METRICS`` except ``'procrustes'`` (default ``'cosine'``).
    - **initialization_strategy (str)**: A key of ``POPULATION_STRATEGIES`` (default ``'random'``).
    - **crossover_type (str)**: A key of ``CROSSOVER_TYPES`` (default ``'uniform'``).
    - **tournament_size (int)**: Tournament size for ``'tournament'`` selection (default 5).
    - **heuristic_population (pd.DataFrame | None)**: Seed for ``'heuristic'``/``'hybrid'``, one
      ``(row, row_m, column, value)`` row per missing cell (default None).
    - **ss_number (int)**: Individuals replaced per generation by ``'steady_state'`` (default 1).
    - **validation_fraction, patience, standardize, full_data_statistics, mutation_type,
      mutation_scale, result_method, elite_size, representation, n_jobs**: see :class:`~GAMIPY.config.GAConfig`.
    - **random_state (int | None)**: Seed of the construction and of ``run()`` (default None: not
      seeded). The same seed gives the same imputation; see :class:`~GAMIPY.config.GAConfig`.

    Attributes
    ----------
    - **imputed_data (pd.DataFrame | None)**: The imputed DataFrame, set by ``run()``.
    - **runner (GeneticAlgorithmRunner)**: Runs the GA over cell values.
    - **model (ModelSearch | None)**: Runs the GA over prediction models (``representation='model'``).
    """
    def __init__(
                self,
                data: pd.DataFrame,
                population_size: int = 100,
                generations: int = 50,
                crossover_rate: float = 0.1,
                mutation_rate: float = 0.01,
                selection_method: str = 'elitist',
                selection_number: int = 10,
                fitness_method: str = 'cosine',
                initialization_strategy: str = 'random',
                crossover_type:str = 'uniform',
                tournament_size: int = 5,
                heuristic_population: pd.DataFrame | None = None,
                ss_number: int = 1,
                validation_fraction: float = 0.1,
                patience: int = 10,
                standardize: bool = True,
                full_data_statistics: bool = True,
                mutation_type: str = 'gaussian',
                mutation_scale: float = 0.1,
                result_method: str = 'best',
                elite_size: int = 5,
                representation: str = 'model',
                n_jobs: int | None = None,
                random_state: int | None = None) -> None:
        self.config = GAConfig(
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            selection_method=selection_method,
            selection_number=selection_number,
            fitness_method=fitness_method,
            initialization_strategy=initialization_strategy,
            crossover_type=crossover_type,
            tournament_size=tournament_size,
            heuristic_population=heuristic_population,
            ss_number=ss_number,
            validation_fraction=validation_fraction,
            patience=patience,
            standardize=standardize,
            full_data_statistics=full_data_statistics,
            mutation_type=mutation_type,
            mutation_scale=mutation_scale,
            result_method=result_method,
            elite_size=elite_size,
            representation=representation,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        if representation not in ('cells', 'model'):
            raise ValueError(f"representation must be 'cells' or 'model', got {representation!r}.")
        thread_count(n_jobs)  # raises for an invalid value
        self.factory = StrategyFactory(self.config)
        self.data = Data(data)
        # The cells GA draws its validation cells when it is built, so the
        # seed covers construction as well as run().
        with _seeded(random_state):
            self.runner = GeneticAlgorithmRunner(self.config, self.data, self.factory)
            # representation='model': the GA evolves per-column prediction models.
            self.model = ModelSearch(self.config, self.data, self.factory) if representation == 'model' else None
        self.imputed_data: pd.DataFrame | None = None

    @property
    def population(self) -> Population | None:
        return self.runner.population

    def initialize_population(self, pop_size: int = 0) -> None:
        self.runner.initialize_population(pop_size)

    def _initialized_population(self) -> Population:
        if self.population is None:
            raise ValueError("Call initialize_population() or run() first.")
        return self.population

    @property
    def selection(self) -> Selection:
        return self.factory.selection(self._initialized_population())

    @property
    def crossover(self) -> Crossover:
        return self.factory.crossover(self._initialized_population().population)

    @property
    def fitness_metric(self) -> DistanceMetric:
        return self.factory.fitness_metric()

    def impute_data(self, individual: Individual) -> pd.DataFrame:
        return self.runner.build_result(individual)

    def run(self) -> None:
        with _seeded(self.config.random_state):
            if self.model is not None:
                self.imputed_data = self.model.run()
                return
            best_individual = self.runner.run()
            self.imputed_data = self.impute_data(best_individual)


@contextmanager
def _seeded(random_state: int | None) -> Iterator[None]:
    """
    Seeds NumPy's and Python's global random number generators with
    `random_state` for the duration of the block, then restores their
    previous states. The GA draws from both; does nothing for None.
    """
    if random_state is None:
        yield
        return
    numpy_state, python_state = np.random.get_state(), random.getstate()
    np.random.seed(random_state)
    random.seed(random_state)
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


if __name__ == "__main__":
    mock_data = sample_data.SampleData.mock_data1()
    iris_data = sample_data.SampleData().load_data("iris")
    gamipy = GAMIPY(data=mock_data, population_size=100, generations=5, crossover_rate=0.2, mutation_rate=0.6)
    gamipy.run()

