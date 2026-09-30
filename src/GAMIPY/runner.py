import warnings
import weakref
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from GAMIPY.bio.encoding import CategoricalEncoder
from GAMIPY.config import GAConfig
from GAMIPY.data.data import Data
from GAMIPY.data.missing_index import MissingIndexLocator
from GAMIPY.imputation import ImputedDataBuilder
from GAMIPY.population.gene import Gene
from GAMIPY.population.individual import Individual
from GAMIPY.population.population import Population
from GAMIPY.strategy_factory import StrategyFactory


class GeneticAlgorithmRunner:

    """
    Runs the GA loop over one dataset and returns the best individual found.
    Split out of GAMIPY, which used to own the loop alongside the
    hyperparameters, the strategy factories and result construction.

    Everything the GA uses comes from the data and the GA itself: no other
    imputation method seeds, scores or replaces its result.

    Attributes
    ----------
    - **config (GAConfig)**: The hyperparameters.
    - **factory (StrategyFactory)**: Builds the population and operators from `config`.
    - **original_data (Data)**: The dataset as given, missing values included.
    - **data (Data)**: The dataset the GA works on: ``original_data`` plus, with
      ``config.validation_fraction > 0``, some observed cells hidden as validation cells.
    - **m_data (Data)**: The rows of `data` with any missing value, index reset.
    - **c_data (Data)**: The complete rows of `data`.
    - **missing_indices (list)**: `(row, row_m, column)` per missing cell, see MissingIndexLocator.
    - **population (Population)**: The current generation, set by `initialize_population`.
    - **validation (dict)**: True value of each validation cell, keyed by ``(row label, column)``.
    - **generations_run (int)**: Generations evaluated by the last ``run()`` (fewer than
      ``config.generations`` after an early stop).
    - **best_validation_error (float | None)**: Validation error of the individual ``run()``
      returned, when validation cells are used.
    - **history (list)**: One dict per generation of the last ``run()``: ``generation``,
      ``best_score`` (lowest fitness score so far) and ``best_validation_error`` (lowest
      validation error so far; NaN without validation cells).
    """

    # Children from crossover (2) + the mutated individual (1), added to the
    # survivors every generation.
    OFFSPRING_COUNT = 3
    # Batches of random individuals tried when the configured strategy can't
    # supply enough distinct individuals, before falling back to duplicates.
    MAX_FILL_ROUNDS = 10

    def __init__(self, config: GAConfig, data: Data, factory: StrategyFactory | None = None) -> None:
        self.config = config
        self.factory = factory if factory is not None else StrategyFactory(config)
        self.original_data = data
        self.validation: dict[tuple[Any, Any], Any] = {}
        self._validation_positions: list[tuple[int, int, Any]] = []
        self.data = self._hide_validation_cells(data) if config.validation_fraction > 0 else data
        # Computed from the full working data, not m_data below, so
        # Gene.orig_location ends up pointing at the user's real DataFrame
        # rows rather than m_data's own reset-index positions.
        self.missing_indices = MissingIndexLocator(self.data).locate()
        # Position of each missing cell, (row label, column), in missing_indices.
        self._cell_order = {(row, col): i for i, (row, _, col) in enumerate(self.missing_indices)}
        self._validation_errors: weakref.WeakKeyDictionary[Individual, float] = weakref.WeakKeyDictionary()
        self.m_data = Data(self.data.get_missing_data)
        self.c_data = Data(self.data.get_complete_data)
        # One category list per column for every encoding, so the codes of
        # the imputed rows and of the complete rows are comparable.
        self.encoder = CategoricalEncoder.for_data(data)
        self.scales = {
            col: float(self.data.get_column(col).std() or 1.0) for col in self.data.get_numeric_columns
        }
        self._fitness_context: dict[str, Any] | None = None
        self.generations_run = 0
        self.best_validation_error: float | None = None
        self.history: list[dict[str, float]] = []
        self.population: Population | None = None
        self._user_seed: pd.DataFrame | None = None

    def _hide_validation_cells(self, data: Data) -> Data:
        """
        Hides ``config.validation_fraction`` of the observed cells (drawn with
        NumPy's global generator, so ``np.random.seed`` makes it reproducible),
        keeping at least two observed values per column. The GA imputes them
        like any missing cell; their true values measure how well it does.
        """
        frame = data.data
        observed = np.argwhere(frame.notna().to_numpy())
        count = round(self.config.validation_fraction * len(observed))
        if count < 1:
            return data
        remaining = frame.notna().sum()
        hidden = frame.copy()
        for pick in np.random.choice(len(observed), size=min(count, len(observed)), replace=False):
            row, col = observed[pick].tolist()
            name = frame.columns[col]
            if remaining[name] <= 2:
                continue
            remaining[name] -= 1
            truth = frame.iat[row, col]
            hidden.iat[row, col] = np.nan
            self.validation[(frame.index[row], name)] = truth
            self._validation_positions.append((row, col, truth))
        return Data(hidden)

    def validation_error(self, individual: Individual) -> float:
        """
        Mean error of `individual` on the validation cells: squared
        standardized difference for numerical cells, 0/1 disagreement for
        categorical ones. NaN when there are no validation cells.
        """
        # Fast path: an individual with one gene per missing cell, in
        # missing_indices order (every individual the GA builds), has the
        # gene for cell c at position _cell_order[c]; only those genes are
        # visited, in gene order, so the sum is the same as the loop below.
        cells = self.validation
        genes = individual.genes
        if len(genes) == len(self._cell_order):
            spots = sorted((self._cell_order[key], key) for key in cells if key in self._cell_order)
            if all(genes[index].location == key for index, key in spots):
                return self._sum_errors([(genes[index], cells[key]) for index, key in spots])
        return self._sum_errors([
            (gene, cells[(gene.location[0], gene.location[1])])
            for gene in genes if (gene.location[0], gene.location[1]) in cells
        ])

    def _cached_validation_error(self, individual: Individual) -> float:
        """`validation_error`, computed once per individual: a survivor's genes
        don't change between generations."""
        error = self._validation_errors.get(individual)
        if error is None:
            error = self._validation_errors[individual] = self.validation_error(individual)
        return error

    def _sum_errors(self, pairs: list[tuple[Gene, Any]]) -> float:
        """Mean error over (gene, true value) pairs; NaN for none."""
        total = 0.0
        count = 0
        for gene, truth in pairs:
            if gene.dtype == 'num':
                scale = self.scales.get(gene.name) or 1.0
                total += ((float(gene.data) - float(truth)) / scale) ** 2
            else:
                total += 0.0 if gene.data == truth else 1.0
            count += 1
        return total / count if count else float('nan')

    def fitness_context(self) -> dict[str, Any]:
        """What Fitness needs from the data (built once): the shared encoder
        and, for ``full_data_statistics``, the observed data to compare with."""
        if self._fitness_context is None:
            context: dict[str, Any] = {'encoder': self.encoder}
            if self.config.full_data_statistics:
                context['reference'] = self.encoder.encode(self.data)
            self._fitness_context = context
        return self._fitness_context

    def _elite_mean(self, elite: list[Individual]) -> Individual:
        """One individual whose numerical genes average `elite`'s and whose
        categorical genes take their most common value (ties: the first
        individual's)."""
        values: dict[Any, list[Any]] = {}
        for individual in elite:
            for gene in individual.genes:
                values.setdefault(gene._m_location, []).append(gene.data)
        first = elite[0]
        mean = Individual(first.id, missing_indices=first.missing_indices, data=first.data)
        mean.genes = [
            Gene(orig_location=gene.location, m_location=gene._m_location, name=gene.name, dtype=gene.dtype,
                 data=float(np.mean(values[gene._m_location])) if gene.dtype == 'num'
                 else Counter(values[gene._m_location]).most_common(1)[0][0])
            for gene in first.genes
        ]
        return mean

    def build_result(self, individual: Individual) -> pd.DataFrame:
        """The original DataFrame with `individual`'s genes filled in and every
        validation cell restored to its true value."""
        result = ImputedDataBuilder(self.data).build(individual)
        for row, col, truth in self._validation_positions:
            result.iat[row, col] = truth
        return result

    def _aligned_seed(self) -> pd.DataFrame:
        """
        The user's ``heuristic_population`` re-aligned to ``missing_indices``:
        one row per missing cell, in the order every other individual uses
        (crossover pairs genes by position). Cells the seed doesn't cover -
        e.g. validation cells - get a random value, drawn the way
        RandomIndividual draws them.
        """
        if self._user_seed is None:
            given = self.config.heuristic_population
            assert given is not None
            values = {(row[0], row[2]): row[3] for row in given.itertuples(index=False)}
            uncovered = [cell for cell in self.missing_indices if (cell[0], cell[2]) not in values]
            if uncovered:
                sampler = Individual(-1, missing_indices=self.missing_indices, data=self.m_data)
                sampler.fallback_data = self.data
                for (row, _, col), value in zip(uncovered, sampler.random_values(uncovered), strict=True):
                    values[(row, col)] = value
            self._user_seed = pd.DataFrame([
                (row, row_m, col, values[(row, col)]) for row, row_m, col in self.missing_indices
            ])
        return self._user_seed

    def _new_population(self, size: int, strategy: str | None = None) -> Population:
        # Values are drawn from m_data (the rows with any missing value), with
        # the full data as a fallback for columns m_data has no observed
        # value in - e.g. a dataset with a single missing cell.
        options = {}
        if self.config.heuristic_population is not None:
            options = {'heuristic_population': self._aligned_seed()}
        population = self.factory.population(size, self.m_data, self.missing_indices, fallback_data=self.data,
                                              strategy=strategy, **options)
        population.initialize()
        return population

    @staticmethod
    def _unique(individuals: list[Individual], taken: set[tuple[Any, ...]]) -> list[Individual]:
        """Individuals whose gene values aren't in `taken` (first occurrence
        wins), adding their keys to `taken`."""
        unique = []
        for individual in individuals:
            key = individual.gene_key()
            if key not in taken:
                taken.add(key)
                unique.append(individual)
        return unique

    def initialize_population(self, pop_size: int = 0, exclude: set[tuple[Any, ...]] | None = None) -> Population:
        """
        Builds `pop_size` (default: `config.population_size`) individuals with
        the configured strategy, all with distinct gene values and none in
        `exclude` (a set of `Individual.gene_key()` values). Duplicates - e.g.
        the identical individuals of the zeros/ones/constant strategies - are
        replaced by random individuals. If the data can't supply enough
        distinct individuals (e.g. every missing cell's column has a single
        observed value), a RuntimeWarning is issued and duplicates fill the
        rest, so the population size never changes.
        """

        size = pop_size if pop_size > 0 else self.config.population_size
        taken = set(exclude) if exclude is not None else set()
        population = self._new_population(size)
        chosen = self._unique(population.population, taken)
        for _ in range(self.MAX_FILL_ROUNDS):
            if len(chosen) >= size:
                break
            extra = self._new_population(size - len(chosen), strategy='random')
            new = self._unique(extra.population, taken)
            if not new:
                break  # the random draws are exhausted; more rounds won't help
            chosen += new
        if len(chosen) < size:
            warnings.warn(
                f"Only {len(chosen)} distinct individuals could be generated for {size} slots; "
                "filling the rest with duplicates (the missing cells have too few possible values).",
                RuntimeWarning, stacklevel=2,
            )
            pool = population.population
            chosen += [pool[i % len(pool)] for i in range(size - len(chosen))]
        population.population = chosen[:size]
        self.population = population
        return population

    def run(self) -> Individual:

        """
        Evolves up to `config.generations` generations and returns the
        individual with the lowest score seen in any of them - or, with
        validation cells, the one with the lowest validation error (or the
        mean of the `config.elite_size` best, see `config.result_method`),
        stopping early after `config.patience` generations without
        improvement.

        Raises
        ------
        - **ValueError**: If `config.generations` is less than 1, selection returns fewer than 2
          individuals, or any strategy name or `config.result_method` is invalid.
        """

        if self.config.generations < 1:
            raise ValueError("generations must be at least 1 to produce a best individual.")
        if self.config.result_method not in ('best', 'elite_mean'):
            raise ValueError(f"result_method must be 'best' or 'elite_mean', got {self.config.result_method!r}.")
        population_size = self.config.population_size
        population = self.initialize_population()
        crossover = self.factory.crossover(population.population)
        mutation = self.factory.mutation(population, self.data)

        best_individual: Individual | None = None
        best_validation: tuple[float, Individual] | None = None
        # The `elite_size` distinct individuals with the lowest validation
        # error seen so far, as (error, gene key, individual).
        elite: list[tuple[float, tuple[Any, ...], Individual]] = []
        # Only kept for 'elite_mean': it compares whole gene tuples.
        track_elite = self.config.result_method == 'elite_mean'
        since_improvement = 0
        self.generations_run = 0
        self.history = []

        for _ in range(self.config.generations):
            # Survivors keep their score: their genes and the context are unchanged.
            fitness = self.factory.fitness(population, self.m_data, self.c_data, **self.fitness_context())
            fitness.evaluate(rescore=False)
            self.generations_run += 1

            selection = self.factory.selection(population)
            selected = selection.select()
            current_best = selection.best_individual
            if best_individual is None or current_best.score < best_individual.score:
                best_individual = current_best

            if self.validation:
                improved = False
                for individual in population.population:
                    error = self._cached_validation_error(individual)
                    if best_validation is None or error < best_validation[0]:
                        best_validation = (error, individual)
                        improved = True
                    if track_elite:
                        elite = self._update_elite(elite, individual, error)
                since_improvement = 0 if improved else since_improvement + 1
            self.history.append({
                'generation': self.generations_run,
                'best_score': float(best_individual.score),
                'best_validation_error': best_validation[0] if best_validation is not None else float('nan'),
            })
            if self.validation and self.config.patience and since_improvement >= self.config.patience:
                break

            # Selection.select() guarantees a non-empty list, but crossover
            # needs two parents.
            if len(selected) < 2:
                raise ValueError(
                    f"Selection returned {len(selected)} individual(s); at least 2 "
                    "are needed as crossover parents (increase selection_number)."
                )
            child1, child2 = crossover.run(selected[0], selected[1])
            mutated = mutation.mutate(selected[-1])

            # Size the next generation from what select actually returned, not
            # selection_number: SteadyState returns population_size - ss_number
            # individuals. Survivors are capped so survivors + offspring never
            # exceed population_size (select returns them sorted, so the cap
            # drops from the tail, same as SteadyState itself).
            survivors = selected[:max(0, population_size - self.OFFSPRING_COUNT)]
            # Every individual in a generation is distinct: selection with
            # replacement can pick the same survivor twice, and offspring can
            # equal a parent. First occurrence wins, so the best survivor stays.
            taken: set[tuple[Any, ...]] = set()
            next_generation = self._unique([*survivors, child1, child2, mutated], taken)
            fill = population_size - len(next_generation)
            if fill > 0:
                population = self.initialize_population(pop_size=fill, exclude=taken)
                population.population = next_generation + population.population
            else:
                population.population = next_generation
            self.population = population

        assert best_individual is not None  # generations >= 1 guarantees one
        if best_validation is None:
            return best_individual
        self.best_validation_error = best_validation[0]
        if self.config.result_method == 'elite_mean' and elite:
            return self._elite_mean([individual for _, _, individual in elite])
        return best_validation[1]

    def _update_elite(
        self, elite: list[tuple[float, tuple[Any, ...], Individual]], individual: Individual, error: float,
    ) -> list[tuple[float, tuple[Any, ...], Individual]]:
        """`elite` with `individual` added if it is among the `elite_size`
        distinct best by validation error. The gene tuple is only built for
        an individual that would make the cut and isn't already there."""
        size = self.config.elite_size
        if len(elite) >= size and not error < elite[-1][0]:
            return elite
        if any(individual is member for _, _, member in elite):
            return elite
        key = individual.gene_key()
        if any(key == other for _, other, _ in elite):
            return elite
        return sorted([*elite, (error, key, individual)], key=lambda e: e[0])[:size]
