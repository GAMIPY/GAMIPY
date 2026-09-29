import unittest

import pandas as pd

from GAMIPY.data.data import Data
from GAMIPY.data.missing_index import MissingIndexLocator
from GAMIPY.population.individual import (
    ConstantIndividual,
    HeuristicIndividual,
    Individual,
    OnesIndividual,
    RandomIndividual,
    ZerosIndividual,
)


class TestRandomIndividual(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({
            'A': [None,2,3],
            'B': [4,5,6],
            'C': ["No","Yes",None]
        }))
        self.missing_indices = MissingIndexLocator(self.data).locate()
        self.random_individual = RandomIndividual(id=1,missing_indices=self.missing_indices, data=self.data)

        self.random_individual.individual()

    def test_missing_indices(self):

        self.assertEqual(self.missing_indices, [(0, 0, 'A'), (2, 1, 'C')])



    def test_random_individual(self):
        individual = self.random_individual
        self.assertEqual(individual.missing_indices, self.missing_indices)
        self.assertEqual(individual.data, self.data)
        self.assertEqual(individual.id, 1)

    def test_get_score(self):
        individual = self.random_individual
        score = individual.get_score
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)

    def test_set_score(self):

        individual = self.random_individual
        individual.set_score(10.0)
        self.assertEqual(individual.get_score, 10.0)

    def test_individual(self):
        individual = self.random_individual
        self.assertIsInstance(individual.genes, list)
        self.assertEqual(individual.genes[0].dtype, "num")
        self.assertEqual(individual.genes[1].dtype, "cat")
        self.assertEqual(len(individual.genes), 2)

    def test_categorical_gene_stores_real_label_not_encoded_int(self):
        # Column 'C' only ever contains "No"/"Yes"/None, so its missing-cell
        # gene must hold one of the real string labels - not an encoded int
        # (CategoricalCodec.encode's 0/1), which is what would end up
        # written back onto the original DataFrame otherwise.
        categorical_gene = self.random_individual.genes[1]
        self.assertIn(categorical_gene.data, ["No", "Yes"])

class TestHeuristicIndividual(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({
            'A': [None,2,3],
            'B': [4,5,6],
            'C': ["No","Yes",None]
        }))

        self.missing_indices = MissingIndexLocator(self.data).locate()
        self.heuristic_data= pd.DataFrame({
            0:[0,2],
            1:[1,2],
            2:['A','C'],
            3:[10,"Maybe"]
        })

        self.heuristic_individual = HeuristicIndividual(
            id=2, missing_indices=self.missing_indices, data=self.data, heuristic_population=self.heuristic_data
        )

    def test_heuristic_individual(self):
        individual = self.heuristic_individual
        self.assertEqual(individual.missing_indices, self.missing_indices)
        self.assertEqual(individual.data, self.data)
        self.assertEqual(individual.id, 2)

    def test_individual(self):
        individual = self.heuristic_individual
        individual.individual()

        self.assertIsInstance(individual.genes, list)

        self.assertEqual(len(individual.genes), 2)

    def test_get_score(self):
        individual = self.heuristic_individual
        score = individual.get_score
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)

    def test_set_score(self):

        individual = self.heuristic_individual
        individual.set_score(10.0)
        self.assertEqual(individual.get_score, 10.0)


class TestConstantIndividual(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({"A": [None, 2, 3], "B": [4, None, 6], "C": ["No", "Yes", "Yes"]}))
        self.missing_indices = MissingIndexLocator(self.data).locate()
        self.random_individual = ConstantIndividual(
            id=1, missing_indices=self.missing_indices, data=self.data, constant_value=42
        )

        self.random_individual.individual()

    def test_missing_indices(self):

        self.assertEqual(self.missing_indices, [(0, 0, "A"), (1, 1, "B")])

    def test_random_individual(self):
        individual = self.random_individual
        self.assertEqual(individual.missing_indices, self.missing_indices)
        self.assertEqual(individual.data, self.data)
        self.assertEqual(individual.id, 1)

    def test_get_score(self):
        individual = self.random_individual
        score = individual.get_score
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)

    def test_set_score(self):

        individual = self.random_individual
        individual.set_score(10.0)
        self.assertEqual(individual.get_score, 10.0)

    def test_individual(self):
        individual = self.random_individual
        self.assertIsInstance(individual.genes, list)
        self.assertEqual(individual.genes[0].dtype, "num")
        self.assertEqual(individual.genes[0].data, 42)
        self.assertEqual(individual.genes[1].dtype, "num")
        self.assertEqual(individual.genes[1].data, 42)
        self.assertEqual(len(individual.genes), 2)

    def test_categorical_missing_cell_gets_the_mode_not_the_constant(self):
        # SetUp above never has a missing value in the
        # categorical column 'C', so the categorical branch of
        # ConstantIndividual.individual() was never exercised by any test -
        # the old bug (dtype="num" hardcoded even for categorical genes,
        # writing an out-of-domain constant like 42 into a "No"/"Yes" column)
        # went undetected. See population/individual.py's ConstantIndividual.
        data = Data(
            pd.DataFrame(
                {
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                    "C": ["No", "Yes", None],
                }
            )
        )
        missing_indices = MissingIndexLocator(data).locate()
        individual = ConstantIndividual(id=1, missing_indices=missing_indices, data=data, constant_value=42)
        individual.individual()

        self.assertEqual(len(individual.genes), 1)
        categorical_gene = individual.genes[0]
        self.assertEqual(categorical_gene.name, "C")
        self.assertEqual(categorical_gene.dtype, "cat")
        self.assertIn(categorical_gene.data, ["No", "Yes"])
        self.assertNotEqual(categorical_gene.data, 42)


class TestZerosIndividual(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({"A": [None, 2, 3], "B": [4, None, 6], "C": ["No", "Yes", "Yes"]}))
        self.missing_indices = MissingIndexLocator(self.data).locate()
        self.random_individual = ZerosIndividual(id=1, missing_indices=self.missing_indices, data=self.data)

        self.random_individual.individual()

    def test_missing_indices(self):

        self.assertEqual(self.missing_indices, [(0, 0, "A"), (1, 1, "B")])

    def test_random_individual(self):
        individual = self.random_individual
        self.assertEqual(individual.missing_indices, self.missing_indices)
        self.assertEqual(individual.data, self.data)
        self.assertEqual(individual.id, 1)

    def test_get_score(self):
        individual = self.random_individual
        score = individual.get_score
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)

    def test_set_score(self):

        individual = self.random_individual
        individual.set_score(10.0)
        self.assertEqual(individual.get_score, 10.0)

    def test_individual(self):
        individual = self.random_individual
        self.assertIsInstance(individual.genes, list)
        self.assertEqual(individual.genes[0].dtype, "num")
        self.assertEqual(individual.genes[0].data, 0.0)
        self.assertEqual(individual.genes[1].dtype, "num")
        self.assertEqual(individual.genes[1].data, 0.0)
        self.assertEqual(len(individual.genes), 2)

    def test_categorical_missing_cell_gets_the_mode_not_zero(self):
        # See TestConstantIndividual.test_categorical_missing_cell_gets_the_mode_not_the_constant.
        data = Data(
            pd.DataFrame(
                {
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                    "C": ["No", "Yes", None],
                }
            )
        )
        missing_indices = MissingIndexLocator(data).locate()
        individual = ZerosIndividual(id=1, missing_indices=missing_indices, data=data)
        individual.individual()

        self.assertEqual(len(individual.genes), 1)
        categorical_gene = individual.genes[0]
        self.assertEqual(categorical_gene.name, "C")
        self.assertEqual(categorical_gene.dtype, "cat")
        self.assertIn(categorical_gene.data, ["No", "Yes"])
        self.assertNotEqual(categorical_gene.data, 0.0)


class TestOnesIndividual(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({"A": [None, 2, 3], "B": [4, None, 6], "C": ["No", "Yes", "Yes"]}))
        self.missing_indices = MissingIndexLocator(self.data).locate()
        self.random_individual = OnesIndividual(id=1, missing_indices=self.missing_indices, data=self.data)

        self.random_individual.individual()

    def test_missing_indices(self):

        self.assertEqual(self.missing_indices, [(0, 0, "A"), (1, 1, "B")])

    def test_random_individual(self):
        individual = self.random_individual
        self.assertEqual(individual.missing_indices, self.missing_indices)
        self.assertEqual(individual.data, self.data)
        self.assertEqual(individual.id, 1)

    def test_get_score(self):
        individual = self.random_individual
        score = individual.get_score
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)

    def test_set_score(self):

        individual = self.random_individual
        individual.set_score(10.0)
        self.assertEqual(individual.get_score, 10.0)

    def test_individual(self):
        individual = self.random_individual
        self.assertIsInstance(individual.genes, list)
        self.assertEqual(individual.genes[0].dtype, "num")
        self.assertEqual(individual.genes[0].data, 1.0)
        self.assertEqual(individual.genes[1].dtype, "num")
        self.assertEqual(individual.genes[1].data, 1.0)
        self.assertEqual(len(individual.genes), 2)

    def test_categorical_missing_cell_gets_the_mode_not_one(self):
        # See TestConstantIndividual.test_categorical_missing_cell_gets_the_mode_not_the_constant.
        data = Data(
            pd.DataFrame(
                {
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                    "C": ["No", "Yes", None],
                }
            )
        )
        missing_indices = MissingIndexLocator(data).locate()
        individual = OnesIndividual(id=1, missing_indices=missing_indices, data=data)
        individual.individual()

        self.assertEqual(len(individual.genes), 1)
        categorical_gene = individual.genes[0]
        self.assertEqual(categorical_gene.name, "C")
        self.assertEqual(categorical_gene.dtype, "cat")
        self.assertIn(categorical_gene.data, ["No", "Yes"])
        self.assertNotEqual(categorical_gene.data, 1.0)


class TestIndividualEvaluationState(unittest.TestCase):
    def setUp(self):
        self.data = Data(
            pd.DataFrame(
                {
                    "A": [None, 2, 3],
                    "B": [4, 5, 6],
                }
            )
        )
        self.individual = RandomIndividual(
            id=1, missing_indices=MissingIndexLocator(self.data).locate(), data=self.data
        )

    def test_new_individual_is_unevaluated_until_scored(self):
        self.assertEqual(self.individual.score, Individual.UNEVALUATED_SCORE)
        self.assertFalse(self.individual.is_evaluated)
        self.individual.set_score(0.0)
        self.assertTrue(self.individual.is_evaluated)
