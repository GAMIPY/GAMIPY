import math
import random
from abc import ABC, abstractmethod
from typing import Any

from GAMIPY.population.individual import Individual
from GAMIPY.population.population import Population


class Selection(ABC):

    """
    A base class for selection strategies in genetic algorithms.

    Attributes
    ----------
    - **selection_number (int)**: The number of individuals to be selected. Default is 10.
    - **population (Population)**: The population from which individuals will be selected. It should
      be an instance of the Population class.

    Methods
    -------
    - **select**: Selects individuals from the population based on the specified selection strategy.
    - **best_individual**: Returns the best individual from the population based on the lowest (distance-based) score.
    - **create**: Builds the strategy registered under a method name, passing each one only the
      extra options it declares.
    """

    # Names of the extra constructor arguments a subclass needs beyond
    # (selection_number, population), e.g. Tournament's tournament_size.
    # Selection.create passes each strategy only the options it declares,
    # so callers never need to know which strategy takes what.
    options: tuple[str, ...] = ()

    @staticmethod
    def create(method: str, selection_number: int, population: Population, **options: Any) -> "Selection":

        """
        Builds the selection strategy registered under `method`.

        Args
        ----
        - **method (str)**: A key of SELECTION_METHODS, e.g. 'elitist' or 'tournament'.
        - **selection_number (int)**: The number of individuals to be selected.
        - **population (Population)**: The population from which individuals will be selected.
        - **options**: Strategy-specific settings (e.g. tournament_size, ss_number). Only the ones
          the chosen strategy declares in its `options` are passed on; the rest are ignored.

        Raises
        ------
        - **ValueError**: If `method` is not a registered selection method.
        """

        try:
            strategy = SELECTION_METHODS[method]
        except KeyError:
            raise ValueError(
                "Invalid selection method. Choose from "
                f"{', '.join(sorted(SELECTION_METHODS))}."
            ) from None
        own_options = {name: options[name] for name in strategy.options if name in options}
        return strategy(selection_number=selection_number, population=population, **own_options)

    def __init__(self, selection_number: int = 10, population: Population | None = None):

        """
        Initializes the Selection object with the specified selection number and population.

        Args
        ----
        - **selection_number (int)**: The number of individuals to be selected. Default is 10.
        - **population (Population)**: The population from which individuals will be selected. It
          should be an instance of the Population class.
        """

        if population is None:
            raise ValueError("Selection needs a population to select from.")
        self.population: Population = population
        self.selection_number:int = selection_number



    @abstractmethod
    def select(self) -> list[Individual]:

        """
        Selects individuals from the population based on the specified selection strategy.

        Contract (every subclass must honour it, so callers such as
        GAMIPY.run() can swap strategies freely):

        - Always returns a non-empty ``list[Individual]``, never ``None``.
        - Raises ``ValueError`` when selection cannot be performed (empty or
          too-small population, unevaluated scores, ...), instead of
          returning a short or empty list.
        - The list holds exactly ``selection_number`` individuals, except for
          SteadyState, which returns the survivors
          (``len(population) - ss_number``). Callers must therefore size
          anything derived from the result by ``len(selected)``, not by
          ``selection_number``.

        Returns
        -------
        - **list[Individual]**: A list of selected individuals based on the selection strategy

        Raises
        ------
        - **ValueError**: If selection cannot be performed on the current population.
        """

        pass

    @property
    def best_individual(self) -> Individual:

        """
        Returns the best individual from the population based on the lowest score.

        Fitness._calculate_score is distance-based (0 = imputed data
        statistically identical to the complete data), so a lower score is a
        better match, not a higher one.

        Returns
        -------
        - **Individual**: The individual with the lowest score in the population. If no best
          individual has been set, it calculates the best individual from the current population.
        """


        best_individual = min(self.population.population, key=lambda ind: ind.score)
        return best_individual

    def _weighted_individuals(self) -> tuple[list[Individual], list[float]]:

        """
        Returns the individuals Fitness has already scored, with a selection
        weight for each, for the fitness-proportional strategies (roulette,
        stochastic universal).

        Scores are distances (lower = better, see best_individual), so they
        cannot be used as weights directly: each individual gets
        ``1 / (1 + score)`` instead - positive, larger for better
        individuals, and never a division by zero. Unevaluated individuals
        hold `Individual.UNEVALUATED_SCORE` and are left out of the draw.

        Raises
        ------
        - **ValueError**: If no individual has been evaluated, or a score is negative or not finite.
        """

        evaluated = [ind for ind in self.population.population if ind.is_evaluated]
        if not evaluated:
            raise ValueError(
                "No evaluated individuals in the population, cannot perform fitness-proportional selection."
            )
        if not all(math.isfinite(ind.score) and ind.score >= 0 for ind in evaluated):
            raise ValueError("Fitness-proportional selection needs finite, non-negative (distance) scores.")
        return evaluated, [1.0 / (1.0 + ind.score) for ind in evaluated]


class Elitist(Selection):
    """
    Implements elitist selection strategy.

    Attributes
    ----------
    - **selection_number (int)**: The number of individuals to be selected. Default is 10.
    - **population (Population)**: The population from which individuals will be selected. It should
      be an instance of the Population class.

    Methods
    -------
    - **select**: Selects individuals from the population based on the specified selection strategy.
    """

    def select(self) -> list[Individual]:
        """
        Performs elitist selection on the population.

        Returns
        -------
        - **list[Individual]**: A list containing the best individual based on the elitist selection strategy.

        Raises
        ------
        - **ValueError**: If the population has fewer than selection_number individuals, indicating
          that selection cannot be performed.
        """

        if len(self.population.population) < self.selection_number:
            raise ValueError("Population is smaller than selection_number, cannot perform elitist selection.")
        # sort the population and return the individual selection number

        # Lower score = better (distance-based fitness), so the elite are
        # the lowest-scoring individuals.
        self.population.population.sort(key=lambda ind: ind.score)
        return self.population.population[: self.selection_number]


class RouletteWheel(Selection):
    """
    Implements roulette wheel selection strategy.

    Attributes
    ----------
    - **selection_number (int)**: The number of individuals to be selected. Default is 10.
    - **population (Population)**: The population from which individuals will be selected. It should
      be an instance of the Population class.

    Methods
    -------
    - **select**: Selects individuals from the population based on the specified selection strategy.
    """

    def select(self) -> list[Individual]:
        """
        Performs roulette wheel selection on the population.

        Returns
        -------
        - **list[Individual]**: A list containing one selected individual based on the roulette
          wheel selection strategy.

        Raises
        ------
        - **ValueError**: If no individual has been evaluated yet, or the total score of the
          evaluated individuals is zero or not finite, indicating that selection cannot be performed.
        """

        # Each individual's slice of the wheel is its weight (1 / (1 + score),
        # see _weighted_individuals), so better (lower-scoring) individuals
        # are more likely to be picked.
        evaluated, weights = self._weighted_individuals()
        total_weight = sum(weights)
        selected = []
        for _ in range(self.selection_number):
            pick = random.uniform(0, total_weight)
            current = 0.0
            for ind, weight in zip(evaluated, weights, strict=True):
                current += weight
                if current > pick:
                    selected.append(ind)
                    break
            else:
                # pick landed exactly on total_weight (uniform's upper bound
                # is inclusive); still return selection_number individuals.
                selected.append(evaluated[-1])

        return selected


class Tournament(Selection):
    options = ("tournament_size",)

    """
    Implements tournament selection strategy.

    Attributes
    ----------
    - **selection_number (int)**: The number of individuals to be selected. Default is 10.
    - **population (Population)**: The population from which individuals will be selected. It should
      be an instance of the Population class.
    - **tournament_size (int)**: The number of individuals to participate in each tournament. Default is 3.

    Methods
    -------
    - **select**: Selects individuals from the population based on the specified selection strategy.
    """

    def __init__(
        self, selection_number: int = 10, tournament_size: int = 3, population: Population | None = None
    ) -> None:
        """
        Initializes the Tournament selection strategy object.

        Args:
        - **selection_number (int)**: The number of individuals to be selected. Default is 10.
        - **tournament_size (int)**: The number of individuals to participate in each tournament. Default is 3.
        - **population (Population)**: The population from which individuals will be selected. It
          should be an instance of the Population class.

        """

        super().__init__(selection_number, population)
        self.tournament_size = tournament_size

    def select(self) -> list[Individual]:
        """
        Performs tournament selection on the population.

        Returns
        -------
        - **list[Individual]**: A list containing one selected individual based on the tournament selection strategy.

        Raises
        ------
        - **ValueError**: If the population size is smaller than the tournament size, indicating
          that a tournament cannot be performed.
        """

        if len(self.population.population) < self.tournament_size:
            raise ValueError("Population size is smaller than tournament size.")
        selected = []
        for _ in range(self.selection_number):
            tournament = random.sample(self.population.population, self.tournament_size)
            winner = min(tournament, key=lambda ind: ind.score)  # lower = better
            selected.append(winner)
        return selected


class Rank(Selection):
    """
    Implements rank selection strategy.

    Attributes
    ----------
    - **selection_number (int)**: The number of individuals to be selected. Default is 10.
    - **population (Population)**: The population from which individuals will be selected. It should
      be an instance of the Population class.

    Methods
    -------
    - **select**: Selects individuals from the population based on the specified selection strategy.
    """

    def select(self) -> list[Individual]:
        """
        Performs rank selection on the population.

        Returns
        -------
        - **list[Individual]**: A list containing one selected individual based on the rank selection strategy.

        """

        if not self.population.population:
            raise ValueError("Population is empty, cannot perform rank selection.")
        # Best (lowest score) first, so it gets the largest rank weight.
        ranked_population = sorted(self.population.population, key=lambda ind: ind.score)
        total_rank = sum(range(1, len(ranked_population) + 1))
        selected = []
        for _ in range(self.selection_number):
            pick = random.uniform(0, total_rank)
            current = 0.0
            for i, ind in enumerate(ranked_population):
                current += len(ranked_population) - i
                if current > pick:
                    selected.append(ind)
                    break
            else:
                # pick landed exactly on total_rank (uniform's upper bound
                # is inclusive); still return selection_number individuals.
                selected.append(ranked_population[-1])
        return selected


class SteadyState(Selection):
    options = ("ss_number",)

    """
    Implements steady state selection strategy.

    Attributes
    ----------
    - **selection_number (int)**: The number of individuals to be selected. Default is 10.
    - **population (Population)**: The population from which individuals will be selected. It should
      be an instance of the Population class.
    - **ss_number (int)**: The number of individuals to be removed from the population. Default is 1.

    Methods
    -------
    - **select**: Selects individuals from the population based on the specified selection strategy.
    """

    def __init__(self, selection_number: int = 10, ss_number: int = 1, population: Population | None = None) -> None:
        """
        Initializes the steady state selection strategy.

        Args:
        - **selection_number (int)**: The number of individuals to be selected. Default is 10.
        - **ss_number (int)**: The number of individuals to be removed from the population. Default is 1.
        - **population (Population)**: The population from which individuals will be selected. It
          should be an instance of the Population class.

        Raises:
        - **ValueError**: If ss_number is smaller than 1 (``population[:-0]`` would silently drop everyone).
        """

        super().__init__(selection_number, population)
        if ss_number < 1:
            raise ValueError("ss_number must be at least 1 for steady state selection.")
        self.ss_number = ss_number

    def select(self) -> list[Individual]:
        """
        Performs steady state selection on the population.

        Unlike the other strategies, this returns the survivors
        (``len(population) - ss_number`` individuals), not
        ``selection_number`` - see the contract on Selection.select().
        GAMIPY.run() sizes the next generation from len(selected).

        Returns
        -------
        - **list[Individual]**: A non-empty list of individuals remaining in the population after
          steady state selection strategy.

        Raises
        ------
        - **ValueError**: If the population has no more than ss_number individuals, so no one would survive.
        """

        if len(self.population.population) <= self.ss_number:
            raise ValueError("Population must be larger than ss_number, cannot perform steady state selection.")
        # Best (lowest score) first, so the ss_number removed are the worst.
        # A new list: the population is left whole, because the runner may
        # stop right after selection (early stopping) and keep it as the last
        # generation - it used to come back ss_number individuals short.
        ranked = sorted(self.population.population, key=lambda ind: ind.score)
        return ranked[: -self.ss_number]


class StochasticUniversal(Selection):
    """
    Implements stochastic universal selection strategy.

    Attributes
    ----------
    - **selection_number (int)**: The number of individuals to be selected. Default is 10.
    - **population (Population)**: The population from which individuals will be selected. It should
      be an instance of the Population class.

    Methods
    -------
    - **select**: Selects individuals from the population based on the specified selection strategy.
    """

    def select(self) -> list[Individual]:
        """
        Performs stochastic universal selection on the population.

        Returns
        -------
        - **list[Individual]**: A list of selected individuals based on the stochastic universal selection strategy.

        Raises
        ------
        - **ValueError**: If no individual has been evaluated yet, or the total score of the
          evaluated individuals is zero or not finite, indicating that selection cannot be performed.
        """

        # Classic SUS: `selection_number` evenly spaced pointers
        # (start + i * distance) laid over the *cumulative* weight, so exactly
        # `selection_number` individuals are always returned, each picked with
        # probability proportional to its weight (1 / (1 + score), see
        # _weighted_individuals). Only individuals already scored by Fitness
        # take part in the draw.
        evaluated, weights = self._weighted_individuals()
        distance = sum(weights) / self.selection_number
        start = random.uniform(0, distance)
        selected = []
        cumulative = 0.0
        index = -1
        last_index = len(evaluated) - 1
        for i in range(self.selection_number):
            pointer = start + i * distance
            # Advance to the individual whose cumulative-weight interval
            # contains the pointer. The index bound guards against float
            # rounding leaving the last pointer just past the final sum.
            while cumulative <= pointer and index < last_index:
                index += 1
                cumulative += weights[index]
            selected.append(evaluated[index])
        return selected


# Method name -> strategy, used by Selection.create. Registering a new
# strategy here (with its own `options`) is all GAMIPY needs to support it.
SELECTION_METHODS: dict[str, type[Selection]] = {
    "elitist": Elitist,
    "tournament": Tournament,
    "roulette": RouletteWheel,
    "rank": Rank,
    "steady_state": SteadyState,
    "stochastic_universal": StochasticUniversal,
}
