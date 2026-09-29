import unittest

import pandas as pd

from GAMIPY.data.data import Data
from GAMIPY.population.individual import Individual, RandomIndividual
from GAMIPY.population.population import (
    POPULATION_STRATEGIES,
    ConstantPopulation,
    HeuristicPopulation,
    HybridPopulation,
    OnesPopulation,
    Population,
    RandomPopulation,
    ZerosPopulation,
)


class TestRandomPopulation(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({
            'A': [None,2,3],
            'B': [4,5,6],
            'C': ["No","Yes",None]
        }))
        self.population_size = 10
        self.random_population = RandomPopulation(data=self.data, population_size=self.population_size)

        self.population = self.random_population.initialize()

    def test_random_population(self):
        self.assertEqual(self.random_population.data, self.data)
        self.assertEqual(self.random_population.population_size, 10)

    def test_individual(self):
        individual = self.random_population.population
        self.assertIsInstance(individual, list)
        self.assertTrue(all(isinstance(ind, Individual) for ind in individual))
        self.assertEqual(len(individual), self.population_size)

    def test_initialize_returns_population_list(self):
        self.assertEqual(self.population, self.random_population.population)

    def test_clear_population(self):
        self.random_population.clear_population()
        self.assertEqual(len(self.random_population.population), 0)

    def test_add_individual(self):
        self.random_population.clear_population()
        individual = RandomIndividual(id=1, missing_indices=[0,1], data=self.data)
        self.random_population.add_individual(individual)
        self.assertEqual(len(self.random_population.population), 1)
        self.assertIsInstance(self.random_population.population[0], Individual)

class TestHeuristicPopulation(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({
            'A': [None,2,3],
            'B': [4,5,6],
            'C': ["No","Yes",None]
        }))
        self.population_size = 10
        self.heuristic_data= pd.DataFrame({
            0:[0,2],
            1:[1,2],
            2:['A','C'],
            3:[10,"Maybe"]
        })

        self.heuristic_population = HeuristicPopulation(
            data=self.data, population_size=self.population_size, heuristic_population=self.heuristic_data
        )
        self.population = self.heuristic_population.initialize()

    def test_random_population(self):
        self.assertEqual(self.heuristic_population.data, self.data)
        self.assertEqual(self.heuristic_population.population_size, 10)

    def test_individual(self):
        individual = self.heuristic_population.population
        self.assertIsInstance(individual, list)
        self.assertTrue(all(isinstance(ind, Individual) for ind in individual))
        self.assertEqual(len(individual), self.population_size)

    def test_initialize_returns_population_list(self):
        self.assertEqual(self.population, self.heuristic_population.population)


class TestZerosPopulation(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({"A": [None, 2, 3], "B": [4, 5, 6], "C": ["No", "Yes", None]}))
        self.population_size = 10
        self.zeros_population = ZerosPopulation(data=self.data, population_size=self.population_size)
        self.population = self.zeros_population.initialize()

    def test_zeros_population(self):
        self.assertEqual(self.zeros_population.data, self.data)
        self.assertEqual(self.zeros_population.population_size, 10)

    def test_individual(self):
        individual = self.zeros_population.population
        self.assertIsInstance(individual, list)
        self.assertTrue(all(isinstance(ind, Individual) for ind in individual))
        self.assertEqual(len(individual), self.population_size)

    def test_initialize_returns_population_list(self):
        self.assertEqual(self.population, self.zeros_population.population)


class TestOnesPopulation(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({"A": [None, 2, 3], "B": [4, 5, 6], "C": ["No", "Yes", None]}))
        self.population_size = 10
        self.ones_population = OnesPopulation(data=self.data, population_size=self.population_size)
        self.population = self.ones_population.initialize()

    def test_ones_population(self):
        self.assertEqual(self.ones_population.data, self.data)
        self.assertEqual(self.ones_population.population_size, 10)

    def test_individual(self):
        individual = self.ones_population.population
        self.assertIsInstance(individual, list)
        self.assertTrue(all(isinstance(ind, Individual) for ind in individual))
        self.assertEqual(len(individual), self.population_size)

    def test_initialize_returns_population_list(self):
        self.assertEqual(self.population, self.ones_population.population)


class TestConstantPopulation(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({"A": [None, 2, 3], "B": [4, 5, 6], "C": ["No", "Yes", None]}))
        self.population_size = 10
        self.constant_population = ConstantPopulation(data=self.data, population_size=self.population_size, constant=42)
        self.population = self.constant_population.initialize()

    def test_constant_population(self):
        self.assertEqual(self.constant_population.data, self.data)
        self.assertEqual(self.constant_population.population_size, 10)

    def test_individual(self):
        individual = self.constant_population.population
        self.assertIsInstance(individual, list)
        self.assertTrue(all(isinstance(ind, Individual) for ind in individual))
        self.assertEqual(len(individual), self.population_size)

    def test_initialize_returns_population_list(self):
        self.assertEqual(self.population, self.constant_population.population)


class TestHybridPopulation(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({"A": [None, 2, 3], "B": [4, 5, 6], "C": ["No", "Yes", None]}))
        self.population_size = 10
        self.heuristic_data = pd.DataFrame({0: [0, 2], 1: [1, 2], 2: ["A", "C"], 3: [10, "Maybe"]})
        self.hybrid_population = HybridPopulation(
            data=self.data, population_size=self.population_size, heuristic_population=self.heuristic_data
        )
        self.population = self.hybrid_population.initialize()

    def test_hybrid_population(self):
        self.assertEqual(self.hybrid_population.data, self.data)
        self.assertEqual(self.hybrid_population.population_size, 10)

    def test_individual(self):
        individual = self.hybrid_population.population
        self.assertIsInstance(individual, list)
        self.assertTrue(any(isinstance(ind, Individual) for ind in individual))
        self.assertTrue(any(isinstance(ind, Individual) for ind in individual))
        self.assertEqual(len(individual), self.population_size)

    def test_initialize_returns_population_list(self):
        self.assertEqual(self.population, self.hybrid_population.population)

    def test_heuristic_population_larger_than_population_size_is_capped(self):
        # When heuristic_population has MORE rows than
        # population_size, num_random used to go negative and initialize
        # raised a size-mismatch ValueError. The seeded share is now capped
        # at population_size (see TestHybridPopulationMoreCellsThanIndividuals).
        larger_heuristic_data = pd.DataFrame(
            {
                0: [0, 2, 4, 6, 8],
                1: [1, 2, 3, 4, 5],
                2: ["A", "C", "B", "D", "E"],
                3: [10, "Maybe", 1, 2, 3],
            }
        )
        oversized_hybrid_population = HybridPopulation(
            data=self.data, population_size=3, heuristic_population=larger_heuristic_data
        )
        self.assertEqual(len(oversized_hybrid_population.initialize()), 3)


class TestHybridPopulationMoreCellsThanIndividuals(unittest.TestCase):
    """heuristic_population has one row per missing cell; with at least as
    many cells as population_size, num_random used to go negative and
    initialize raised ValueError."""

    def setUp(self):
        from GAMIPY.data.missing_index import MissingIndexLocator

        df = pd.DataFrame({"A": [1.0, None, None, None, 5.0], "B": [None, 2.0, None, 4.0, None]})
        self.data = Data(df)
        self.seed = pd.DataFrame(
            [(row, row_m, col, 3.0) for row, row_m, col in MissingIndexLocator(self.data).locate()]
        )  # 6 missing cells -> 6 rows
        self.missing_indices = MissingIndexLocator(self.data).locate()

    def build(self, population_size):
        population = HybridPopulation(
            population_size=population_size,
            data=self.data,
            heuristic_population=self.seed,
            missing_indices=self.missing_indices,
        )
        population.initialize()
        return population.population

    def test_fewer_individuals_than_missing_cells(self):
        for size in (1, 3, 6):
            with self.subTest(population_size=size):
                self.assertEqual(len(self.build(size)), size)

    def test_half_seeded_half_random_whatever_the_number_of_cells(self):
        # 10 individuals, 6 cells: 5 from the seed family (the seed itself +
        # 4 variants), 5 random - no longer "as many seeds as missing cells".
        population = self.build(10)
        self.assertEqual(len(population), 10)
        exact_seed = [ind for ind in population if all(g.data == 3.0 for g in ind.genes)]
        self.assertEqual(len(exact_seed), 1)
        variants = population[1:5]
        for variant in variants:
            # Each variant keeps most of the seed but differs in at least one gene.
            self.assertGreaterEqual(sum(g.data == 3.0 for g in variant.genes), 1)
            self.assertNotEqual(variant.gene_key(), population[0].gene_key())


class TestHeuristicPopulationVariants(unittest.TestCase):
    """HeuristicPopulation = the seed + distinct variants of it (identical
    copies add nothing to the search)."""

    def test_seed_first_then_distinct_variants(self):
        from GAMIPY.data.missing_index import MissingIndexLocator

        df = pd.DataFrame({"A": [1.0, None, 3.0, None, 5.0], "B": [None, 2.0, 7.0, 4.0, None]})
        data = Data(df)
        missing = MissingIndexLocator(data).locate()
        seed = pd.DataFrame([(row, row_m, col, 3.0) for row, row_m, col in missing])
        population = HeuristicPopulation(
            population_size=8, data=data, heuristic_population=seed, missing_indices=missing
        )
        individuals = population.initialize()
        self.assertEqual(len(individuals), 8)
        self.assertTrue(all(g.data == 3.0 for g in individuals[0].genes))
        for variant in individuals[1:]:
            self.assertNotEqual(variant.gene_key(), individuals[0].gene_key())
            self.assertEqual([g._m_location for g in variant.genes], [g._m_location for g in individuals[0].genes])


class TestPopulationCreate(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({"A": [1.0, None, 3.0], "B": [4.0, 5.0, None]}))
        self.heuristic_population = pd.DataFrame({0: [1.5, 2.5], 1: [4.5, 5.5]})

    def test_create_builds_each_registered_strategy(self):
        for name, cls in POPULATION_STRATEGIES.items():
            with self.subTest(name=name):
                population = Population.create(
                    name,
                    population_size=4,
                    data=self.data,
                    heuristic_population=self.heuristic_population,
                    constant=0.5,
                )
                self.assertIsInstance(population, cls)
                self.assertEqual(population.population_size, 4)

    def test_create_passes_only_declared_options(self):
        # RandomPopulation declares no options, so extras are dropped rather
        # than raising TypeError from its constructor.
        population = Population.create("random", population_size=4, data=self.data, constant=0.5)
        self.assertIsInstance(population, RandomPopulation)
        population = Population.create("constant", population_size=4, data=self.data, constant=0.25)
        self.assertEqual(population.constant, 0.25)

    def test_create_with_unknown_strategy_raises_value_error(self):
        with self.assertRaises(ValueError):
            Population.create("not_a_real_strategy", population_size=4, data=self.data)
