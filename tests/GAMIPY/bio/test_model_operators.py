"""The Bio-module operators of the model-based GA (representation='model'):
ModelFitness, ColumnTournament, ArithmeticModelCrossover and
GaussianModelMutation, and how ModelSearch gets them from StrategyFactory."""
import unittest

import numpy as np
import pandas as pd

from GAMIPY.bio.crossover import ArithmeticModelCrossover
from GAMIPY.bio.fitness import ModelFitness
from GAMIPY.bio.mutation import GaussianModelMutation
from GAMIPY.bio.selection import ColumnTournament
from GAMIPY.config import GAConfig
from GAMIPY.data.data import Data
from GAMIPY.model_search import ModelSearch
from GAMIPY.strategy_factory import StrategyFactory


def _fitness() -> ModelFitness:
    # Columns: x (numerical input), y (numerical target), c (categorical
    # target, 2 classes). Features: x, y, c=0, c=1; outputs: y, c=0, c=1.
    return ModelFitness(targets=['y', 'c'], numeric=['x', 'y'],
                        feature_slots={'x': [0], 'y': [1], 'c': [2, 3]}, output_slots={'y': [0], 'c': [1, 2]})


class TestModelFitness(unittest.TestCase):

    def test_predict_is_linear_without_missing_inputs(self):
        weights = np.zeros((1, 3, 4))
        weights[0, 0, 0] = 2.0  # y = 2x + 1
        bias = np.array([[1.0, 0.0, 0.0]])
        features = np.array([[1.0, 0.0, 1.0, 0.0], [3.0, 0.0, 0.0, 1.0]])
        outputs = _fitness().predict(weights, bias, features, np.zeros((2, 4), dtype=bool))
        np.testing.assert_allclose(outputs[0, :, 0], [3.0, 7.0])

    def test_missing_inputs_take_the_models_predictions(self):
        # z predicts y; w predicts z from y. Row 0's y is missing, so after
        # chaining w sees y's prediction (2) and predicts z = 3 * 2.
        fitness = ModelFitness(targets=['y', 'z'], numeric=['x', 'y', 'z'],
                               feature_slots={'x': [0], 'y': [1], 'z': [2]}, output_slots={'y': [0], 'z': [1]})
        weights = np.zeros((1, 2, 3))
        weights[0, 0, 0] = 2.0  # y = 2x
        weights[0, 1, 1] = 3.0  # z = 3y
        features = np.array([[1.0, 0.0, 0.0]])
        missing = np.array([[False, True, False]])
        outputs = fitness.predict(weights, np.zeros((1, 2)), features, missing)
        self.assertAlmostEqual(outputs[0, 0, 1], 6.0)

    def test_numerical_error_is_the_mean_squared_error(self):
        outputs = np.zeros((1, 2, 3))
        outputs[0, :, 0] = [1.0, 3.0]
        cells = {'y': (np.array([0, 1]), np.array([0.0, 1.0]))}
        errors = _fitness().column_errors(outputs, cells, 'log')
        self.assertAlmostEqual(errors[0, 0], (1.0 + 4.0) / 2)
        self.assertTrue(np.isnan(errors[0, 1]))  # no cells for c

    def test_categorical_losses(self):
        outputs = np.zeros((1, 1, 3))
        outputs[0, 0, 1:] = [2.0, 0.0]  # class 0 scores higher
        right = {'c': (np.array([0]), np.array([0.0]))}
        wrong = {'c': (np.array([0]), np.array([1.0]))}
        fitness = _fitness()
        self.assertEqual(fitness.column_errors(outputs, right, 'zero_one')[0, 1], 0.0)
        self.assertEqual(fitness.column_errors(outputs, wrong, 'zero_one')[0, 1], 1.0)
        self.assertLess(fitness.column_errors(outputs, right, 'log')[0, 1],
                        fitness.column_errors(outputs, wrong, 'log')[0, 1])

    def test_evaluate_matches_for_any_number_of_threads(self):
        rng = np.random.default_rng(0)
        weights, bias = rng.normal(size=(40, 3, 4)), rng.normal(size=(40, 3))
        features = rng.normal(size=(20, 4))
        missing = np.zeros((20, 4), dtype=bool)
        cells = {'y': (np.arange(20), rng.normal(size=20))}
        single = _fitness().evaluate(weights, bias, features, missing, cells, {})
        threaded = ModelFitness(targets=['y', 'c'], numeric=['x', 'y'],
                                feature_slots={'x': [0], 'y': [1], 'c': [2, 3]},
                                output_slots={'y': [0], 'c': [1, 2]}, n_jobs=3).evaluate(
            weights, bias, features, missing, cells, {})
        np.testing.assert_array_equal(single[0], threaded[0])
        np.testing.assert_array_equal(single[0], single[1])  # no validation cells: judged = training


class TestColumnTournament(unittest.TestCase):

    def test_winner_has_the_lowest_score_of_its_entrants(self):
        np.random.seed(0)
        scores = np.array([5.0, 1.0, 3.0, 4.0])
        self.assertEqual(ColumnTournament(tournament_size=4).select(scores), 1)

    def test_tournament_larger_than_population_uses_everyone(self):
        np.random.seed(0)
        self.assertEqual(ColumnTournament(tournament_size=10).select(np.array([2.0, 0.5])), 1)


class TestArithmeticModelCrossover(unittest.TestCase):

    def test_rate_zero_copies_the_first_parent(self):
        w, b = ArithmeticModelCrossover(0.0).cross(np.ones(3), np.ones(1), np.zeros(3), np.zeros(1))
        np.testing.assert_array_equal(w, np.ones(3))
        np.testing.assert_array_equal(b, np.ones(1))

    def test_rate_one_gives_a_weighted_average_of_the_parents(self):
        np.random.seed(0)
        w, b = ArithmeticModelCrossover(1.0).cross(np.full(3, 2.0), np.full(1, 2.0), np.zeros(3), np.zeros(1))
        self.assertTrue(np.all((w >= 0) & (w <= 2)))
        self.assertTrue(np.allclose(w, w[0]))  # one share for the whole model
        self.assertAlmostEqual(b[0], w[0])


class TestGaussianModelMutation(unittest.TestCase):

    def test_rate_zero_changes_nothing(self):
        weights, bias = np.ones((2, 3)), np.ones(2)
        w, b = GaussianModelMutation(0.0, 1.0).mutate(weights, bias, np.ones((2, 3)))
        np.testing.assert_array_equal(w, weights)
        np.testing.assert_array_equal(b, bias)

    def test_disallowed_weights_stay_zero(self):
        np.random.seed(0)
        allowed = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
        w, _ = GaussianModelMutation(1.0, 1.0).mutate(np.ones((2, 3)), np.ones(2), allowed)
        self.assertEqual(w[0, 0], 0.0)
        self.assertEqual(w[1, 1], 0.0)
        self.assertFalse(np.allclose(w[allowed == 1], 1.0))

    def test_inputs_are_not_modified(self):
        np.random.seed(0)
        weights, bias = np.ones((2, 3)), np.ones(2)
        GaussianModelMutation(1.0, 1.0).mutate(weights, bias, np.ones((2, 3)))
        np.testing.assert_array_equal(weights, np.ones((2, 3)))


class TestModelSearchUsesBioOperators(unittest.TestCase):

    def test_operators_come_from_the_config(self):
        config = GAConfig(tournament_size=3, crossover_rate=0.7, mutation_rate=0.2, mutation_scale=0.05)
        df = pd.DataFrame({'a': [1.0, 2.0, None, 4.0, 5.0], 'b': [2.0, 4.0, 6.0, None, 10.0]})
        search = ModelSearch(config, Data(df), StrategyFactory(config))
        self.assertIsInstance(search.fitness, ModelFitness)
        self.assertEqual(search.selection.tournament_size, 3)
        self.assertEqual(search.crossover.crossover_rate, 0.7)
        self.assertEqual(search.mutation.mutation_rate, 0.2)
        self.assertEqual(search.mutation.mutation_scale, 0.05)
        self.assertEqual(search.fitness.targets, ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
