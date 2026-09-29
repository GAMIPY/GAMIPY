import unittest

import pandas as pd

from GAMIPY.bio.mutation import Mutation
from GAMIPY.bio.value_sampling import (
    CategoricalValueSampler,
    NumericalValueSampler,
    ValueSampler,
)
from GAMIPY.data.categorical import CategoricalColumn
from GAMIPY.data.data import Data
from GAMIPY.data.numerical import NumericalColumn
from GAMIPY.population.individual import Individual
from GAMIPY.population.population import RandomPopulation


class TestMutation(unittest.TestCase):
    def setUp(self):
        self.mutation_rate = 0.1
        self.data = Data(pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, None],
            'C': [None, "No", "Yes"]
        }))

        self.random_population = RandomPopulation(data=self.data, population_size=10)
        self.random_population.initialize()
        self.mutation = Mutation(
            mutation_rate=self.mutation_rate, population=self.random_population.population, data=self.data
        )

    def test_initialization(self):
        self.assertEqual(self.mutation.mutation_rate, self.mutation_rate)
        self.assertEqual(self.mutation.population, self.random_population.population)
        self.assertEqual(self.mutation.data, self.data)

    def test_mutate(self):
        mutated_individual = self.mutation.mutate(self.random_population.population[0])
        self.assertIsNotNone(mutated_individual)
        self.assertIsInstance(mutated_individual, Individual)

    def test_mutate_keeps_values_within_the_column_domain(self):
        # The test above only checked that mutate()
        # returns a non-None Individual, never that a mutated gene's value
        # actually lands in the column's valid domain - a numerical gene
        # within [min_val, max_val], a categorical gene within `classes`.
        # mutation_rate=1.0 forces every gene to mutate so this is checked
        # on every gene, not left to chance.
        mutation = Mutation(mutation_rate=1.0, population=self.random_population.population, data=self.data)
        mutated_individual = mutation.mutate(self.random_population.population[0])

        self.assertGreater(len(mutated_individual.genes), 0)
        for gene in mutated_individual.genes:
            if gene.dtype == "num":
                numerical_column = NumericalColumn(self.data, gene.name)
                self.assertGreaterEqual(gene.data, numerical_column.min_val)
                self.assertLessEqual(gene.data, numerical_column.max_val)
            elif gene.dtype == "cat":
                categorical_column = CategoricalColumn(self.data, gene.name)
                self.assertIn(gene.data, categorical_column.classes)
            else:
                self.fail(f"Unexpected gene dtype: {gene.dtype}")

    def test_mutation_defaults_to_its_own_value_samplers(self):
        self.assertIsInstance(self.mutation.numeric_sampler, NumericalValueSampler)
        self.assertIsInstance(self.mutation.categorical_sampler, CategoricalValueSampler)

    def test_mutate_routes_through_the_injected_samplers(self):
        # Proves mutate() genuinely depends on the injected samplers (not
        # just accepts and ignores them): with mutation_rate=1.0, every gene
        # mutates, and a stub sampler's sentinel should show up on every
        # gene of the matching dtype.
        class StubSampler(ValueSampler):
            def __init__(self, sentinel):
                self.sentinel = sentinel

            def sample(self, data, column):
                return self.sentinel

        mutation = Mutation(
            mutation_rate=1.0,
            population=self.random_population.population,
            data=self.data,
            numeric_sampler=StubSampler("NUM_SENTINEL"),
            categorical_sampler=StubSampler("CAT_SENTINEL"),
        )
        mutated = mutation.mutate(self.random_population.population[0])
        for gene in mutated.genes:
            expected = "NUM_SENTINEL" if gene.dtype == "num" else "CAT_SENTINEL"
            self.assertEqual(gene.data, expected)


class TestMutationLeavesInputUnchanged(unittest.TestCase):
    """mutate() used to change the selected individual in place and return
    the same object, so a survivor (or the best individual so far) could end
    up with genes that no longer matched its score."""

    def test_returns_a_new_unevaluated_individual(self):
        import pandas as pd

        from GAMIPY.data.data import Data
        from GAMIPY.population.population import RandomPopulation

        data = Data(pd.DataFrame({"A": [1.0, None, 3.0, 4.0], "B": [5.0, 6.0, None, 8.0]}))
        population = RandomPopulation(population_size=1, data=data)
        population.initialize()
        parent = population.population[0]
        parent.set_score(0.5)
        before = [(g.name, g.data) for g in parent.genes]

        mutant = Mutation(mutation_rate=1.0, population=population, data=data).mutate(parent)

        self.assertIsNot(mutant, parent)
        self.assertEqual([(g.name, g.data) for g in parent.genes], before)
        self.assertEqual(parent.score, 0.5)
        self.assertFalse(mutant.is_evaluated)
        self.assertTrue(all(m is not p for m, p in zip(mutant.genes, parent.genes, strict=True)))
