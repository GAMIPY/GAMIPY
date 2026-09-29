import random
import unittest
import warnings

import pandas as pd

from GAMIPY.bio.selection import (
    SELECTION_METHODS,
    Elitist,
    Rank,
    RouletteWheel,
    Selection,
    SteadyState,
    StochasticUniversal,
    Tournament,
)
from GAMIPY.data.data import Data
from GAMIPY.population.individual import Individual
from GAMIPY.population.population import RandomPopulation


class TestElitist(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, None, 6],
            'target': [None, 1, 0]
        }))
        self.population = RandomPopulation(population_size=100,data=self.data)
        self.population.initialize()
        self.elitist = Elitist(selection_number=2,population=self.population)

    def test_initialization(self):
        self.assertIsInstance(self.elitist, Elitist)
        self.assertEqual(self.elitist.selection_number, 2)
        self.assertEqual(self.elitist.population, self.population)


    def test_selection(self):
        selected = self.elitist.select()

        self.assertEqual(len(selected), 2)
        self.assertTrue(all(isinstance(ind, Individual) for ind in selected))

    def test_selection_on_population_smaller_than_selection_number_raises_value_error(self):
        # Used to silently return a shorter list than selection_number.
        self.population.population = self.population.population[:1]
        with self.assertRaises(ValueError):
            self.elitist.select()

    def test_best_individual_returns_lowest_scored_individual(self):
        # Fitness._calculate_score is distance-based (0 = perfect match), so
        # best_individual must pick the LOWEST score, not the highest.
        for i, ind in enumerate(self.population.population):
            ind.set_score(float(i))
        expected_best = min(self.population.population, key=lambda ind: ind.score)
        self.assertIs(self.elitist.best_individual, expected_best)
        self.assertEqual(self.elitist.best_individual.score, 0.0)



class TestRouletteWheel(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, None, 6],
            'target': [None, 1, 0]
        }))
        self.population = RandomPopulation(population_size=100,data=self.data)
        self.population.initialize()
        self.elitist = RouletteWheel(selection_number=6,population=self.population)

    def test_initialization(self):
        self.assertIsInstance(self.elitist, RouletteWheel)
        self.assertEqual(self.elitist.selection_number, 6)
        self.assertEqual(self.elitist.population, self.population)


    def test_selection(self):
        for i, ind in enumerate(self.population.population):
            ind.set_score(float(i + 1))
        selected = self.elitist.select()
        self.assertEqual(len(selected), 6)
        self.assertTrue(all(isinstance(ind, Individual) for ind in selected))

    def test_selection_ignores_unevaluated_individuals_without_overflow(self):
        # Only half the population is evaluated; the rest still hold
        # Individual.UNEVALUATED_SCORE. Summing those used to overflow to inf.
        evaluated = self.population.population[:50]
        for i, ind in enumerate(evaluated):
            ind.set_score(float(i + 1))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            selected = self.elitist.select()
        self.assertEqual(len(selected), 6)
        self.assertTrue(all(ind.is_evaluated for ind in selected))

    def test_selection_on_unevaluated_population_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.elitist.select()


class TestTournament(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, None, 6],
            'target': [None, 1, 0]
        }))
        self.population = RandomPopulation(population_size=100,data=self.data)
        self.population.initialize()
        self.elitist = Tournament(selection_number=6,population=self.population)

    def test_initialization(self):
        self.assertIsInstance(self.elitist, Tournament)
        self.assertEqual(self.elitist.selection_number, 6)
        self.assertEqual(self.elitist.population, self.population)


    def test_selection(self):
        selected = self.elitist.select()
        self.assertEqual(len(selected), 6)
        self.assertTrue(all(isinstance(ind, Individual) for ind in selected))

    def test_population_smaller_than_tournament_size_raises_value_error(self):
        self.population.population = self.population.population[:2]
        with self.assertRaises(ValueError):
            self.elitist.select()


class TestRank(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, None, 6],
            'target': [None, 1, 0]
        }))
        self.population = RandomPopulation(population_size=100,data=self.data)
        self.population.initialize()
        self.elitist = Rank(selection_number=6,population=self.population)

    def test_initialization(self):
        self.assertIsInstance(self.elitist, Rank)
        self.assertEqual(self.elitist.selection_number, 6)
        self.assertEqual(self.elitist.population, self.population)


    def test_selection(self):
        selected = self.elitist.select()
        self.assertEqual(len(selected), 6)
        self.assertTrue(all(isinstance(ind, Individual) for ind in selected))

    def test_selection_on_empty_population_raises_value_error(self):
        self.population.population = []
        with self.assertRaises(ValueError):
            self.elitist.select()


class TestSteadyState(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, None, 6],
            'target': [None, 1, 0]
        }))
        self.population = RandomPopulation(population_size=100,data=self.data)
        self.population.initialize()
        self.elitist = SteadyState(selection_number=6,population=self.population,ss_number=3)

    def test_initialization(self):
        self.assertIsInstance(self.elitist, SteadyState)
        self.assertEqual(self.elitist.selection_number, 6)
        self.assertEqual(self.elitist.population, self.population)


    def test_selection(self):
        # SteadyState returns the survivors, not selection_number
        # individuals (documented exception in Selection.select()'s contract).
        selected = self.elitist.select()
        self.assertEqual(len(selected), 100 - 3)
        self.assertTrue(all(isinstance(ind, Individual) for ind in selected))

    def test_selection_on_too_small_population_raises_value_error(self):
        # Used to return None, which then crashed at the caller's selected[0].
        self.population.population = self.population.population[:3]
        with self.assertRaises(ValueError):
            self.elitist.select()

    def test_ss_number_below_one_raises_value_error(self):
        with self.assertRaises(ValueError):
            SteadyState(selection_number=6, population=self.population, ss_number=0)


class TestStochasticUniversal(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, None, 6],
            'target': [None, 1, 0]
        }))
        self.population = RandomPopulation(population_size=100,data=self.data)
        self.population.initialize()
        self.elitist = StochasticUniversal(selection_number=6,population=self.population)

    def test_initialization(self):
        self.assertIsInstance(self.elitist, StochasticUniversal)
        self.assertEqual(self.elitist.selection_number, 6)
        self.assertEqual(self.elitist.population, self.population)


    def test_selection(self):
        for i, ind in enumerate(self.population.population):
            ind.set_score(float(i + 1))
        selected = self.elitist.select()
        self.assertEqual(len(selected), 6)
        self.assertTrue(all(isinstance(ind, Individual) for ind in selected))

    def test_selection_uses_cumulative_scores(self):
        # Weights are 1 / (1 + score): scores [0, 2, 2, 2] give weights
        # [1, 1/3, 1/3, 1/3], so the first individual owns half the total
        # weight and, with 6 evenly spaced pointers, must be picked exactly
        # 3 times whatever the random start - the old implementation compared
        # the pointer to each individual's own value instead of the
        # cumulative sum.
        population = self.population.population[:4]
        self.population.population = population
        for ind, score in zip(population, [0.0, 2.0, 2.0, 2.0], strict=True):
            ind.set_score(score)
        for _ in range(50):
            selected = self.elitist.select()
            self.assertEqual(len(selected), 6)
            self.assertEqual(sum(ind is population[0] for ind in selected), 3)
            # SUS returns individuals in population order.
            self.assertEqual(
                [population.index(ind) for ind in selected],
                sorted(population.index(ind) for ind in selected),
            )

    def test_lower_score_gets_proportionally_more_picks(self):
        # Scores [9, 0, 9] -> weights [0.1, 1.0, 0.1] (total 1.2); the 6
        # pointers are 0.2 apart, so the best (score 0) individual's
        # interval of width 1.0 always catches exactly 5 of them.
        population = self.population.population[:3]
        self.population.population = population
        for ind, score in zip(population, [9.0, 0.0, 9.0], strict=True):
            ind.set_score(score)
        for _ in range(50):
            selected = self.elitist.select()
            self.assertEqual(len(selected), 6)
            self.assertEqual(sum(ind is population[1] for ind in selected), 5)

    def test_selection_on_unevaluated_population_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.elitist.select()

    def test_selection_ignores_unevaluated_individuals_without_overflow(self):
        evaluated = self.population.population[:50]
        for i, ind in enumerate(evaluated):
            ind.set_score(float(i + 1))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            selected = self.elitist.select()
        self.assertEqual(len(selected), 6)
        self.assertTrue(all(ind.is_evaluated for ind in selected))

    def test_selection_on_empty_population_raises_value_error(self):
        self.population.population = []
        with self.assertRaises(ValueError):
            self.elitist.select()

class TestSelectionCreate(unittest.TestCase):
    """Selection.create builds a strategy by name and hands each one only
    the options it declares, so callers (GAMIPY.selection) don't need to
    know which strategy takes tournament_size, ss_number, ..."""

    def setUp(self):
        self.data = Data(pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, None, 6],
            'target': [None, 1, 0]
        }))
        self.population = RandomPopulation(population_size=10, data=self.data)
        self.population.initialize()
        self.all_options = {'tournament_size': 4, 'ss_number': 2}

    def test_every_registered_method_builds_its_strategy(self):
        for method, strategy in SELECTION_METHODS.items():
            with self.subTest(method=method):
                selection = Selection.create(method, selection_number=3, population=self.population, **self.all_options)
                self.assertIsInstance(selection, strategy)
                self.assertEqual(selection.selection_number, 3)
                self.assertIs(selection.population, self.population)

    def test_strategies_receive_only_their_own_options(self):
        tournament = Selection.create('tournament', 3, self.population, **self.all_options)
        self.assertEqual(tournament.tournament_size, 4)
        self.assertFalse(hasattr(tournament, 'ss_number'))

        steady_state = Selection.create('steady_state', 3, self.population, **self.all_options)
        self.assertEqual(steady_state.ss_number, 2)
        self.assertFalse(hasattr(steady_state, 'tournament_size'))

        elitist = Selection.create('elitist', 3, self.population, **self.all_options)
        self.assertFalse(hasattr(elitist, 'tournament_size'))
        self.assertFalse(hasattr(elitist, 'ss_number'))

    def test_missing_option_falls_back_to_strategy_default(self):
        tournament = Selection.create('tournament', 3, self.population)
        self.assertEqual(tournament.tournament_size, 3)

    def test_invalid_method_raises_value_error(self):
        with self.assertRaises(ValueError):
            Selection.create('not_a_real_method', 3, self.population)

class TestSelectionPrefersLowerScores(unittest.TestCase):
    """Fitness scores are distances (lower = better). Every strategy must
    favour low scores - before 2026-09-28 they all favoured high ones while
    best_individual took the lowest, so the GA evolved its worst solutions."""

    def setUp(self):
        df = pd.DataFrame({'A': [1.0, None, 3.0, 4.0], 'B': [5.0, 6.0, None, 8.0]})
        self.population = RandomPopulation(population_size=20, data=Data(df))
        self.population.initialize()
        # Score i for the i-th individual: individual 0 is the best.
        for i, ind in enumerate(self.population.population):
            ind.set_score(float(i))
        self.best = self.population.population[0]
        self.worst = self.population.population[-1]

    def scores(self, individuals):
        return [ind.score for ind in individuals]

    def test_elitist_keeps_the_lowest_scores(self):
        selected = Elitist(selection_number=5, population=self.population).select()
        self.assertEqual(sorted(self.scores(selected)), [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_tournament_winner_is_the_lowest_score(self):
        # A tournament over the whole population must pick the best.
        selection = Tournament(selection_number=3, tournament_size=20, population=self.population)
        self.assertEqual(self.scores(selection.select()), [0.0, 0.0, 0.0])

    def test_steady_state_removes_the_highest_scores(self):
        survivors = SteadyState(ss_number=3, population=self.population).select()
        self.assertEqual(max(self.scores(survivors)), 16.0)
        self.assertIn(self.best, survivors)

    def test_best_individual_is_the_lowest_score(self):
        self.assertIs(Elitist(population=self.population).best_individual, self.best)

    def test_stochastic_methods_pick_the_best_more_often_than_the_worst(self):
        random.seed(0)
        for cls in (RouletteWheel, Rank, StochasticUniversal):
            with self.subTest(method=cls.__name__):
                picks = []
                for _ in range(200):
                    picks += cls(selection_number=5, population=self.population).select()
                best_count = sum(ind is self.best for ind in picks)
                worst_count = sum(ind is self.worst for ind in picks)
                self.assertGreater(best_count, worst_count)
