from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class GAConfig:

    """
    The genetic algorithm's hyperparameters, split out of GAMIPY, which used
    to hold them alongside the strategy factories, the GA loop and result
    construction.

    Strategy names are stored as plain strings and only resolved (and
    validated) by StrategyFactory when first needed.

    Attributes
    ----------
    - **population_size (int)**: The number of individuals per generation (default 100).
    - **generations (int)**: The number of generations to run (default 50).
    - **crossover_rate (float)**: The probability of crossover between two parents (default 0.1).
    - **mutation_rate (float)**: The probability of mutating each gene (default 0.01).
    - **selection_method (str)**: A key of `bio.selection.SELECTION_METHODS` (default 'elitist').
    - **selection_number (int)**: The number of individuals to select per generation (default 10).
    - **fitness_method (str)**: A key of `bio.distance.DISTANCE_METRICS`, except 'procrustes' (default 'cosine').
    - **initialization_strategy (str)**: A key of `population.population.POPULATION_STRATEGIES`
      (default 'random').
    - **crossover_type (str)**: A key of `bio.crossover.CROSSOVER_TYPES` (default 'uniform').
    - **tournament_size (int)**: The tournament size for 'tournament' selection (default 5).
    - **heuristic_population (pd.DataFrame)**: Seed individuals for 'heuristic'/'hybrid' initialization (default None).
    - **ss_number (int)**: The number of individuals removed per generation for 'steady_state' selection (default 1).
    - **constant (float)**: The fill value for 'constant' initialization (default 0.5).
    - **validation_fraction (float)**: Share of the observed cells hidden as validation cells
      (default 0.1; 0 turns validation off). The GA imputes them too; the individual that imputes
      them best is returned.
    - **patience (int)**: With validation cells, stop after this many generations without a better
      validation error (default 10; 0: never stop early).
    - **standardize (bool)**: Compare z-scored means, skewness and correlation matrices instead of
      raw means, skewness and covariance matrices (default True).
    - **full_data_statistics (bool)**: Compute the statistics of the whole imputed dataset and
      compare them with the observed data, instead of comparing the imputed rows with the complete
      rows (default True).
    - **mutation_type (str)**: ``'gaussian'`` (default: add noise with ``mutation_scale`` times the
      column's standard deviation, clipped to the observed range) or ``'uniform'`` (resample a
      numerical gene anywhere in the column's observed range).
    - **mutation_scale (float)**: Noise scale for Gaussian mutation (default 0.1).
    - **result_method (str)**: With validation cells, what ``run()`` returns: ``'best'`` (default:
      the individual with the lowest validation error) or ``'elite_mean'`` (the average of the
      ``elite_size`` best, mode for categorical cells).
    - **elite_size (int)**: Individuals averaged by ``'elite_mean'`` (default 5).
    - **representation (str)**: What an individual is: ``'cells'`` (default: one gene per missing
      cell) or ``'model'`` (per-column linear prediction models, see ``GAMIPY.model_search``).

    GAMIPY is a pure genetic algorithm: no other imputation method seeds its
    population, enters its fitness or replaces its result. Other imputers
    belong in benchmarks, as baselines to compare against.
    """

    population_size: int = 100
    generations: int = 50
    crossover_rate: float = 0.1
    mutation_rate: float = 0.01
    selection_method: str = 'elitist'
    selection_number: int = 10
    fitness_method: str = 'cosine'
    initialization_strategy: str = 'random'
    crossover_type: str = 'uniform'
    tournament_size: int = 5
    heuristic_population: pd.DataFrame | None = None
    ss_number: int = 1
    constant: float = 0.5
    validation_fraction: float = 0.1
    patience: int = 10
    standardize: bool = True
    full_data_statistics: bool = True
    mutation_type: str = 'gaussian'
    mutation_scale: float = 0.1
    result_method: str = 'best'
    elite_size: int = 5
    representation: str = 'cells'

    @property
    def selection_options(self) -> dict[str, Any]:
        # Strategy-specific settings, handed to Selection.create as a bag;
        # each strategy takes only the ones it declares in its `options`.
        return {
            'tournament_size': self.tournament_size,
            'ss_number': self.ss_number,
        }

    @property
    def population_options(self) -> dict[str, Any]:
        # Same, for Population.create.
        return {
            'heuristic_population': self.heuristic_population,
            'constant': self.constant,
        }
