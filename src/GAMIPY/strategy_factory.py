from typing import Any

from GAMIPY.bio.crossover import Crossover
from GAMIPY.bio.distance import DistanceMetric
from GAMIPY.bio.fitness import Fitness
from GAMIPY.bio.mutation import Mutation
from GAMIPY.bio.selection import Selection
from GAMIPY.config import GAConfig
from GAMIPY.data.data import Data
from GAMIPY.data.missing_index import MissingCell
from GAMIPY.population.individual import Individual
from GAMIPY.population.population import Population


class StrategyFactory:

    """
    Builds the GA's operators from a GAConfig, by delegating to each
    abstraction's own registry factory (`Population.create`,
    `Selection.create`, `Crossover.create`, `DistanceMetric.create`). Split
    out of GAMIPY so the GA loop depends only on these abstractions, not on
    how config values map onto them.

    The config is read on every call rather than once up front, so a bad
    strategy name raises only when that operator is first needed, and a
    config changed after construction is still honoured.

    Attributes
    ----------
    - **config (GAConfig)**: The hyperparameters to build operators from.
    """

    def __init__(self, config: GAConfig) -> None:
        self.config = config

    def population(
        self,
        population_size: int,
        data: Data,
        missing_indices: list[MissingCell],
        fallback_data: Data | None = None,
        strategy: str | None = None,
        **option_overrides: Any,
    ) -> Population:
        # `strategy` / `option_overrides` override the configured strategy and
        # its options: the runner uses 'random' to top up with distinct
        # individuals, and passes the aligned heuristic seed.
        return Population.create(
            strategy if strategy is not None else self.config.initialization_strategy,
            population_size=population_size,
            data=data,
            missing_indices=missing_indices,
            fallback_data=fallback_data,
            **{**self.config.population_options, **option_overrides},
        )

    def selection(self, population: Population) -> Selection:
        return Selection.create(
            self.config.selection_method,
            selection_number=self.config.selection_number,
            population=population,
            **self.config.selection_options,
        )

    def crossover(self, individuals: list[Individual]) -> Crossover:
        # Crossover.population is documented/typed as `list` (a list of
        # Individuals - see Crossover.__init__ and is_all_numerical()),
        # unlike Selection.population, which expects the Population object
        # itself - so callers pass population.population here.
        return Crossover.create(
            self.config.crossover_type,
            crossover_rate=self.config.crossover_rate,
            population=individuals,
        )

    def fitness_metric(self) -> DistanceMetric:
        return DistanceMetric.create(self.config.fitness_method)

    def fitness(self, population: Population, m_data: Data, c_data: Data, **context: Any) -> Fitness:
        # `context` carries what the runner derives from the data (encoder,
        # reference); the switches come from the config.
        return Fitness(
            population=population,
            m_data=m_data,
            c_data=c_data,
            metric=self.fitness_metric(),
            full_data_statistics=self.config.full_data_statistics,
            standardize=self.config.standardize,
            n_jobs=self.config.n_jobs,
            **context,
        )

    def mutation(self, population: Population, data: Data) -> Mutation:
        return Mutation(
            mutation_rate=self.config.mutation_rate,
            population=population,
            data=data,
            mutation_type=self.config.mutation_type,
            mutation_scale=self.config.mutation_scale,
        )
