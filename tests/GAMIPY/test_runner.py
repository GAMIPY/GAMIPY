import itertools
import unittest

import numpy as np
import pandas as pd

from GAMIPY.bio.crossover import Blend
from GAMIPY.bio.distance import EuclideanDistance
from GAMIPY.config import GAConfig
from GAMIPY.data.data import Data
from GAMIPY.data.missing_index import MissingIndexLocator
from GAMIPY.GAMIPY import GAMIPY
from GAMIPY.imputation import ImputedDataBuilder
from GAMIPY.population.individual import Individual
from GAMIPY.population.population import ZerosPopulation
from GAMIPY.runner import GeneticAlgorithmRunner
from GAMIPY.strategy_factory import StrategyFactory


class TestGAConfig(unittest.TestCase):
    def test_defaults_match_gamipy_defaults(self):
        config = GAConfig()
        self.assertEqual(config.population_size, 100)
        self.assertEqual(config.generations, 50)
        self.assertEqual(config.selection_method, 'elitist')
        self.assertEqual(config.crossover_type, 'uniform')
        self.assertEqual(config.initialization_strategy, 'random')
        self.assertEqual(config.fitness_method, 'cosine')

    def test_strategy_option_bags(self):
        config = GAConfig(tournament_size=4, ss_number=2, constant=0.25)
        self.assertEqual(config.selection_options, {'tournament_size': 4, 'ss_number': 2})
        self.assertEqual(config.population_options, {'heuristic_population': None, 'constant': 0.25})


class TestStrategyFactory(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({'A': [1.0, None, 3.0], 'B': [4.0, 5.0, None]}))

    def test_reads_config_on_every_call(self):
        config = GAConfig(crossover_type='uniform', fitness_method='cosine', initialization_strategy='random')
        factory = StrategyFactory(config)
        config.crossover_type = 'blend'
        config.fitness_method = 'euclidean'
        config.initialization_strategy = 'zeros'
        self.assertIsInstance(factory.crossover([]), Blend)
        self.assertIsInstance(factory.fitness_metric(), EuclideanDistance)
        self.assertIsInstance(factory.population(3, self.data, None), ZerosPopulation)

    def test_invalid_names_raise_value_error(self):
        factory = StrategyFactory(
            GAConfig(crossover_type="nope", fitness_method="nope", initialization_strategy="nope")
        )
        with self.assertRaises(ValueError):
            factory.crossover([])
        with self.assertRaises(ValueError):
            factory.fitness_metric()
        with self.assertRaises(ValueError):
            factory.population(3, self.data, None)


class TestGeneticAlgorithmRunner(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.df = pd.DataFrame(
            {
                "f1": rng.uniform(0, 10, 30),
                "f2": rng.uniform(0, 10, 30),
            }
        )
        self.df.loc[3, "f1"] = None
        self.df.loc[7, "f2"] = None

    def test_run_returns_best_individual_and_keeps_population_size(self):
        config = GAConfig(population_size=20, generations=2, selection_number=5, validation_fraction=0.0)
        runner = GeneticAlgorithmRunner(config, Data(self.df))
        best = runner.run()
        self.assertIsInstance(best, Individual)
        self.assertEqual(len(best.genes), 2)
        self.assertEqual(len(runner.population.population), 20)

    def test_best_score_never_gets_worse_and_improves_on_the_start(self):
        # Records each generation's best score. With elitist selection the
        # best individual always survives, so the sequence can't increase;
        # and a working GA should beat its initial random population.
        import random as _random

        _random.seed(0)
        np.random.seed(0)
        # The classic GA (no validation cells, uniform mutation, plain
        # statistics), where the returned individual is the best by fitness.
        config = GAConfig(
            population_size=30, generations=15, selection_number=8, crossover_rate=0.8, mutation_rate=0.2,
            validation_fraction=0.0, standardize=False, full_data_statistics=False,
            mutation_type='uniform',
        )
        runner = GeneticAlgorithmRunner(config, Data(self.df))
        per_generation = []
        build_fitness = runner.factory.fitness

        def recording_fitness(population, m_data, c_data, **context):
            fitness = build_fitness(population, m_data, c_data, **context)
            evaluate = fitness.evaluate

            def evaluate_and_record(**options):
                evaluate(**options)
                per_generation.append(min(ind.score for ind in population.population))

            fitness.evaluate = evaluate_and_record
            return fitness

        runner.factory.fitness = recording_fitness
        best = runner.run()
        self.assertEqual(len(per_generation), 15)
        for earlier, later in itertools.pairwise(per_generation):
            self.assertLessEqual(later, earlier + 1e-12)
        self.assertLess(per_generation[-1], per_generation[0])
        self.assertAlmostEqual(best.score, per_generation[-1])

    def test_zero_generations_raises_value_error(self):
        runner = GeneticAlgorithmRunner(GAConfig(population_size=20, generations=0), Data(self.df))
        with self.assertRaises(ValueError):
            runner.run()

    def test_uses_an_injected_factory(self):
        config = GAConfig(population_size=20, generations=1, selection_number=5)

        class RecordingFactory(StrategyFactory):
            def __init__(self, config):
                super().__init__(config)
                self.calls = []

            def selection(self, population):
                self.calls.append("selection")
                return super().selection(population)

        factory = RecordingFactory(config)
        GeneticAlgorithmRunner(config, Data(self.df), factory).run()
        self.assertEqual(factory.calls, ["selection"])


class TestUniqueIndividuals(unittest.TestCase):
    """Every individual in a GA population has distinct gene values
    (Individual.gene_key): duplicates from selection with replacement, from
    offspring equal to a parent, or from deterministic strategies
    (zeros/ones/constant) are replaced by new individuals."""

    def setUp(self):
        rng = np.random.default_rng(3)
        self.df = pd.DataFrame({"f1": rng.uniform(0, 10, 30), "f2": rng.uniform(0, 10, 30)})
        self.df.loc[[2, 5, 9], "f1"] = None
        self.df.loc[[5, 12], "f2"] = None

    def keys(self, population):
        return [ind.gene_key() for ind in population.population]

    def test_every_generation_is_distinct_for_every_strategy(self):
        for init in ["random", "zeros", "ones", "constant", "heuristic", "hybrid"]:
            with self.subTest(initialization_strategy=init):
                seed = pd.DataFrame(
                    [(row, row_m, col, 5.0) for row, row_m, col in MissingIndexLocator(Data(self.df)).locate()]
                )
                config = GAConfig(
                    population_size=15,
                    generations=4,
                    selection_number=5,
                    selection_method="roulette",
                    initialization_strategy=init,
                    heuristic_population=seed,
                )
                runner = GeneticAlgorithmRunner(config, Data(self.df))
                initial = runner.initialize_population()
                self.assertEqual(len(set(self.keys(initial))), 15)
                runner.run()
                keys = self.keys(runner.population)
                self.assertEqual(len(keys), 15)
                self.assertEqual(len(set(keys)), 15)

    def test_zeros_strategy_keeps_one_zero_individual_and_fills_with_random(self):
        runner = GeneticAlgorithmRunner(GAConfig(population_size=10, initialization_strategy="zeros"), Data(self.df))
        keys = self.keys(runner.initialize_population())
        self.assertEqual(len(set(keys)), 10)
        self.assertEqual(sum(all(v == 0.0 for v in key) for key in keys), 1)

    def test_degenerate_data_warns_and_keeps_the_population_size(self):
        # Every missing cell's column has a single possible value.
        df = pd.DataFrame({"a": [1.0, 1.0, 1.0, None], "b": [2.0, 3.0, 4.0, 5.0]})
        runner = GeneticAlgorithmRunner(GAConfig(population_size=6, validation_fraction=0.0), Data(df))
        with self.assertWarns(RuntimeWarning):
            population = runner.initialize_population()
        self.assertEqual(len(population.population), 6)


class TestImputedDataBuilder(unittest.TestCase):
    def test_build_fills_every_missing_cell(self):
        df = pd.DataFrame({"f1": [1.0, None, 3.0], "f2": [4.0, 5.0, None]}, index=[10, 11, 12])
        gamipy = GAMIPY(data=df, population_size=3)
        gamipy.initialize_population()
        imputed = ImputedDataBuilder(Data(df)).build(gamipy.population.population[0])
        self.assertFalse(imputed.isnull().any().any())
        self.assertEqual(imputed.loc[10, "f1"], 1.0)


class TestGAMIPYFacade(unittest.TestCase):
    def test_config_changes_after_construction_are_honoured(self):
        df = pd.DataFrame({"A": [1.0, None, 3.0], "B": [4.0, 5.0, 6.0]})
        gamipy = GAMIPY(data=df)
        gamipy.config.fitness_method = "euclidean"
        self.assertIsInstance(gamipy.fitness_metric, EuclideanDistance)


if __name__ == "__main__":
    unittest.main()
