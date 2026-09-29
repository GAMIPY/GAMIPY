import unittest

import pandas as pd

from GAMIPY.bio.crossover import CROSSOVER_TYPES, Arithmetic, Blend, Crossover, OnePoint, TwoPoint, Uniform
from GAMIPY.data.data import Data
from GAMIPY.population.population import RandomPopulation


class TestUniform(unittest.TestCase):

    def setUp(self):
        self.data = Data(
                pd.DataFrame({
                    'feature1': [1, None, 3],
                    'feature2': [4, 5, 6],
                    'feature3': [7, 8, None]
                })
        )
        self.population = RandomPopulation(population_size=10, data=self.data)
        self.population.initialize()
        self.crossover_rate = 0.4

        self.uniform_crossover = Uniform(self.crossover_rate,self.population.population)

    def test_initialization(self):
        self.assertIsInstance(self.uniform_crossover, Crossover)
        self.assertEqual(self.uniform_crossover.crossover_rate, self.crossover_rate)
        self.assertEqual(self.uniform_crossover.population, self.population.population)

    def test_crossover(self):
        self.parent1 = self.population.population[0]
        self.parent2 = self.population.population[1]
        offspring = self.uniform_crossover.run(parent1=self.parent1, parent2=self.parent2)
        self.assertEqual(len(offspring), 2)



class TestOnePoint(unittest.TestCase):

    def setUp(self):
        self.data = Data(
                pd.DataFrame({
                    'feature1': [1, None, 3],
                    'feature2': [4, 5, 6],
                    'feature3': [7, 8, None]
                })
        )
        self.population = RandomPopulation(population_size=10, data=self.data)
        self.population.initialize()
        self.crossover_rate = 0.4

        self.uniform_crossover = OnePoint(self.crossover_rate,self.population.population)

    def test_initialization(self):
        self.assertIsInstance(self.uniform_crossover, Crossover)
        self.assertEqual(self.uniform_crossover.crossover_rate, self.crossover_rate)
        self.assertEqual(self.uniform_crossover.population, self.population.population)

    def test_crossover(self):
        self.parent1 = self.population.population[0]
        self.parent2 = self.population.population[1]
        offspring = self.uniform_crossover.run(parent1=self.parent1, parent2=self.parent2)
        self.assertEqual(len(offspring), 2)

class TestTwoPoint(unittest.TestCase):

    def setUp(self):
        # This class used to be entirely disabled inside a
        # triple-quoted string because the 2-missing-value fixture the other
        # classes in this file use gives each individual only 2 genes, and
        # TwoPoint.run needs >=3 genes for its two cut points
        # (point1 in [1, len-2], point2 in [point1+1, len-1]) - it crashed
        # with "ValueError: empty range for randrange()" every time. This
        # fixture has 3 missing values (one per column) instead.
        self.data = Data(
                pd.DataFrame({
                    'feature1': [1, None, 3, 4],
                    'feature2': [5, 6, None, 8],
                    'feature3': [9, 10, 11, None]
                })
        )
        self.population = RandomPopulation(population_size=10, data=self.data)
        self.population.initialize()
        self.crossover_rate = 0.4

        self.two_point_crossover = TwoPoint(self.crossover_rate,self.population.population)

    def test_initialization(self):
        self.assertIsInstance(self.two_point_crossover, Crossover)
        self.assertEqual(self.two_point_crossover.crossover_rate, self.crossover_rate)
        self.assertEqual(self.two_point_crossover.population, self.population.population)

    def test_crossover(self):
        self.parent1 = self.population.population[0]
        self.parent2 = self.population.population[1]
        offspring = self.two_point_crossover.run(parent1=self.parent1, parent2=self.parent2)
        self.assertEqual(len(offspring), 2)
        child1, child2 = offspring

        self.assertEqual(len(child1.genes), len(self.parent1.genes))
        self.assertEqual(len(child2.genes), len(self.parent2.genes))

        # With exactly 3 genes, the two cut points are forced to point1=1,
        # point2=2 (randint(1, 1) and randint(2, 2) both have only one
        # possible outcome), so only the middle gene is swapped between
        # parents - deterministic, not just "some split happened".
        self.assertEqual(child1.genes[0].data, self.parent1.genes[0].data)
        self.assertEqual(child1.genes[1].data, self.parent2.genes[1].data)
        self.assertEqual(child1.genes[2].data, self.parent1.genes[2].data)

        self.assertEqual(child2.genes[0].data, self.parent2.genes[0].data)
        self.assertEqual(child2.genes[1].data, self.parent1.genes[1].data)
        self.assertEqual(child2.genes[2].data, self.parent2.genes[2].data)

class TestArithmetic(unittest.TestCase):

    def setUp(self):
        self.data = Data(
                pd.DataFrame({
                    'feature1': [1, None, 3],
                    'feature2': [4, 5, 6],
                    'feature3': [7, 8, None]
                })
        )
        self.population = RandomPopulation(population_size=10, data=self.data)
        self.population.initialize()
        self.crossover_rate = 0.4

        self.uniform_crossover = Arithmetic(self.crossover_rate,self.population.population)

    def test_initialization(self):
        self.assertIsInstance(self.uniform_crossover, Crossover)
        self.assertEqual(self.uniform_crossover.crossover_rate, self.crossover_rate)
        self.assertEqual(self.uniform_crossover.population, self.population.population)

    def test_crossover(self):
        self.parent1 = self.population.population[0]
        self.parent2 = self.population.population[1]
        offspring = self.uniform_crossover.run(parent1=self.parent1, parent2=self.parent2)
        self.assertEqual(len(offspring), 2)
        child1, child2 = offspring

        # Checking only len(offspring) == 2 (the tuple)
        # never inspected the children's own gene lists - this is what let a
        # real bug (child1/child2 rebuilt as new empty Individuals on every
        # loop iteration, so only the LAST gene pair survived) go uncaught.
        # Fixed now: one gene per missing value, in the right location, with
        # the actual averaged numeric value.
        self.assertEqual(len(child1.genes), len(self.parent1.genes))
        self.assertEqual(len(child2.genes), len(self.parent2.genes))

        for gene1, gene2, c1_gene, c2_gene in zip(
            self.parent1.genes, self.parent2.genes, child1.genes, child2.genes, strict=True
        ):
            expected_average = (gene1.data + gene2.data) / 2
            for child_gene in (c1_gene, c2_gene):
                self.assertEqual(child_gene.location, gene1.location)
                self.assertEqual(child_gene._m_location, gene1._m_location)
                self.assertEqual(child_gene.dtype, "num")
                self.assertIsInstance(child_gene.data, float)
                self.assertAlmostEqual(child_gene.data, expected_average)

    def test_raises_via_is_all_numerical_for_a_categorical_gene(self):
        # Arithmetic.run() now delegates to the base
        # class's is_all_numerical() instead of duplicating this check
        # inline.
        data = Data(pd.DataFrame({
            'feature1': [1, None, 3],
            'feature2': ['x', 'y', None],
        }))
        population = RandomPopulation(population_size=10, data=data)
        population.initialize()
        crossover = Arithmetic(self.crossover_rate, population.population)
        with self.assertRaises(ValueError):
            crossover.run(population.population[0], population.population[1])


class TestBlend(unittest.TestCase):

    def setUp(self):
        self.data = Data(

                pd.DataFrame({
                    'feature1': [1, None, 3],
                    'feature2': [4, 5, 6],
                    'feature3': [7, 8, None]
                })
        )
        self.population = RandomPopulation(population_size=10, data=self.data)
        self.population.initialize()
        self.crossover_rate = 0.4

        self.uniform_crossover = Blend(self.crossover_rate,self.population.population)

    def test_initialization(self):
        self.assertIsInstance(self.uniform_crossover, Crossover)
        self.assertEqual(self.uniform_crossover.crossover_rate, self.crossover_rate)
        self.assertEqual(self.uniform_crossover.population, self.population.population)

    def test_crossover(self):
        self.parent1 = self.population.population[0]
        self.parent2 = self.population.population[1]
        offspring = self.uniform_crossover.run(parent1=self.parent1, parent2=self.parent2)
        self.assertEqual(len(offspring), 2)
        child1, child2 = offspring

        self.assertEqual(len(child1.genes), len(self.parent1.genes))
        self.assertEqual(len(child2.genes), len(self.parent2.genes))

        for gene1, gene2, c1_gene, c2_gene in zip(
            self.parent1.genes, self.parent2.genes, child1.genes, child2.genes, strict=True
        ):
            self.assertEqual(c1_gene.location, gene1.location)
            self.assertEqual(c1_gene._m_location, gene1._m_location)
            self.assertEqual(c1_gene.dtype, "num")
            self.assertIsInstance(c1_gene.data, float)
            # alpha is random (uniform(0.5, 1.5)), but
            # new_data1 + new_data2 = alpha*g1 + (1-alpha)*g2 + alpha*g2 +
            # (1-alpha)*g1 = g1 + g2 regardless of alpha's value.
            self.assertAlmostEqual(c1_gene.data + c2_gene.data, gene1.data + gene2.data)

    def test_raises_via_is_all_numerical_for_a_categorical_gene(self):
        # Blend.run() now delegates to the base class's
        # is_all_numerical() instead of duplicating this check inline.
        data = Data(pd.DataFrame({
            'feature1': [1, None, 3],
            'feature2': ['x', 'y', None],
        }))
        population = RandomPopulation(population_size=10, data=data)
        population.initialize()
        crossover = Blend(self.crossover_rate, population.population)
        with self.assertRaises(ValueError):
            crossover.run(population.population[0], population.population[1])
class TestFewGenes(unittest.TestCase):
    """Found by tests/GAMIPY/test_integration.py: OnePoint crashed on 1 gene
    and TwoPoint on fewer than 3 (random.randint(1, 0))."""

    def _parents(self, n_missing: int):
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0], 'B': [5.0, 6.0, 7.0, 8.0]})
        cells = [(0, 'A'), (1, 'B'), (2, 'A')][:n_missing]
        for row, col in cells:
            df.loc[row, col] = None
        population = RandomPopulation(population_size=2, data=Data(df))
        population.initialize()
        return population.population[0], population.population[1]

    def test_one_point_with_a_single_gene_copies_parents(self):
        parent1, parent2 = self._parents(1)
        child1, child2 = OnePoint(crossover_rate=1.0).run(parent1, parent2)
        self.assertEqual([g.data for g in child1.genes], [g.data for g in parent1.genes])
        self.assertEqual([g.data for g in child2.genes], [g.data for g in parent2.genes])

    def test_two_point_with_fewer_than_three_genes_still_crosses_over(self):
        for n_missing in (1, 2):
            with self.subTest(n_missing=n_missing):
                parent1, parent2 = self._parents(n_missing)
                child1, child2 = TwoPoint(crossover_rate=1.0).run(parent1, parent2)
                self.assertEqual(len(child1.genes), n_missing)
                self.assertEqual(len(child2.genes), n_missing)


class TestNumericCrossover(unittest.TestCase):
    """is_all_numerical lives on NumericCrossover (Arithmetic, Blend) only,
    and checks the parents being crossed, not the population list the
    crossover was created with (which the GA loop replaces every generation)."""

    def test_only_numeric_crossovers_have_the_check(self):
        from GAMIPY.bio.crossover import NumericCrossover
        self.assertTrue(issubclass(Arithmetic, NumericCrossover))
        self.assertTrue(issubclass(Blend, NumericCrossover))
        for cls in (Uniform, OnePoint, TwoPoint):
            self.assertFalse(hasattr(cls, 'is_all_numerical'))

    def test_parents_are_checked_even_with_an_empty_population(self):
        df = pd.DataFrame({'A': [1.0, None, 3.0], 'C': ['x', 'y', None]})
        population = RandomPopulation(population_size=2, data=Data(df))
        population.initialize()
        parent1, parent2 = population.population
        for cls in (Arithmetic, Blend):
            with self.subTest(crossover=cls.__name__):
                with self.assertRaises(ValueError):
                    cls(crossover_rate=1.0, population=[]).run(parent1, parent2)


class TestCrossoverCreate(unittest.TestCase):
    def test_create_builds_each_registered_type(self):
        for name, cls in CROSSOVER_TYPES.items():
            with self.subTest(name=name):
                crossover = Crossover.create(name, crossover_rate=0.3, population=[])
                self.assertIsInstance(crossover, cls)
                self.assertEqual(crossover.crossover_rate, 0.3)

    def test_create_with_unknown_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            Crossover.create('not_a_real_type', crossover_rate=0.3, population=[])


