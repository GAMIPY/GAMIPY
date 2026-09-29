import unittest

import numpy as np
import pandas as pd

from GAMIPY.bio.column_statistics import (
    CategoricalCentralTendency,
    ColumnStatisticsProvider,
    NumericalCentralTendency,
)
from GAMIPY.bio.distance import CosineDistance, EuclideanDistance
from GAMIPY.bio.encoding import CategoricalEncoder
from GAMIPY.bio.fitness import Fitness
from GAMIPY.data.data import Data
from GAMIPY.population.population import RandomPopulation


class TestFitness(unittest.TestCase):
    def setUp(self):
        self.data = Data(pd.DataFrame({
            'A': [None,2,3],
            'B': [4,5,6],
            'C': ["No","Yes",None]
        }))
        self.random_population = RandomPopulation(data=self.data, population_size=10)
        self.random_population.initialize()
        self.m_data = Data(self.data.get_missing_data)
        self.c_data = self.data.get_complete_data
        self.fitness = Fitness(population=self.random_population.population, m_data=self.m_data, c_data=self.c_data)

    def test_fitness_initialization(self):
        self.assertIsInstance(self.fitness, Fitness)


    def test_impute_data(self):
        imputed_data = self.fitness.impute_data(self.random_population.population[0])
        self.assertIsInstance(imputed_data, pd.DataFrame)
        self.assertEqual(imputed_data.shape, self.data.get_missing_data.shape)

    def test_fitness_defaults_to_its_own_categorical_encoder(self):
        self.assertIsInstance(self.fitness.encoder, CategoricalEncoder)

    def test_fitness_accepts_an_injected_categorical_encoder(self):
        encoder = CategoricalEncoder()
        fitness = Fitness(
            population=self.random_population.population, m_data=self.m_data, c_data=self.c_data, encoder=encoder
        )
        self.assertIs(fitness.encoder, encoder)

    def test_fitness_defaults_to_cosine_metric(self):
        self.assertIsInstance(self.fitness.metric, CosineDistance)

    def test_fitness_accepts_an_injected_metric(self):
        metric = EuclideanDistance()
        fitness = Fitness(
            population=self.random_population.population, m_data=self.m_data, c_data=self.c_data, metric=metric
        )
        self.assertIs(fitness.metric, metric)

    def test_fitness_defaults_to_its_own_column_statistics_providers(self):
        self.assertIsInstance(self.fitness.numeric_stats, NumericalCentralTendency)
        self.assertIsInstance(self.fitness.categorical_stats, CategoricalCentralTendency)

    def test_mean_score_routes_through_the_injected_providers(self):
        # Proves mean_score genuinely depends on the injected providers
        # (not just accepts and ignores them): stub providers that return a
        # fixed sentinel should show up in mean_score's output.
        class StubProvider(ColumnStatisticsProvider):
            def __init__(self, sentinel):
                self.sentinel = sentinel

            def central_value(self, data, column):
                return self.sentinel

        data = Data(pd.DataFrame({"A": [1.0, 2.0], "B": ["x", "y"]}))
        fitness = Fitness(
            numeric_stats=StubProvider("NUMERIC_SENTINEL"),
            categorical_stats=StubProvider("CATEGORICAL_SENTINEL"),
        )
        self.assertEqual(fitness.mean_score(data), ["NUMERIC_SENTINEL", "CATEGORICAL_SENTINEL"])

    def test_covariance_score_matches_pandas_cov(self):
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [4.0, 1.0, 6.0, 2.0]})
        data = Data(df)
        expected = df.cov().to_numpy()
        actual = self.fitness.covariance_score(data)
        self.assertIsInstance(actual, np.ndarray)
        np.testing.assert_allclose(actual, expected)

    def test_skewness_score_matches_pandas_skew(self):
        df = pd.DataFrame({"A": [1.0, 2.0, 2.0, 3.0, 10.0], "B": [4.0, 1.0, 6.0, 2.0, 3.0]})
        data = Data(df)
        expected = df.skew().to_numpy()
        actual = self.fitness.skewness_score(data)
        self.assertIsInstance(actual, np.ndarray)
        np.testing.assert_allclose(actual, expected)

    def test_calculate_score_matches_the_documented_formula(self):
        # The score is a sum of distances (lower = better):
        # metric(mean) + metric(skewness) + procrustes(covariance). Verify
        # that's actually what comes out,
        # using data skewed enough to avoid the all-zero-skewness /
        # divide-by-zero edge case (see test_GAMIPY.py's TestFitnessMetricProperty).
        rng = np.random.default_rng(0)
        m_df = pd.DataFrame({"A": rng.uniform(0, 10, 10), "B": rng.uniform(0, 10, 10)})
        c_df = pd.DataFrame({"A": rng.uniform(0, 10, 10) ** 2, "B": rng.uniform(0, 10, 10)})
        m_data, c_data = Data(m_df), Data(c_df)

        fitness = Fitness(metric=CosineDistance())
        mean_m, mean_c = fitness.mean_score(m_data), fitness.mean_score(c_data)
        skew_m, skew_c = fitness.skewness_score(m_data), fitness.skewness_score(c_data)
        cov_m, cov_c = fitness.covariance_score(m_data), fitness.covariance_score(c_data)

        from GAMIPY.bio.distance import ProcrustesDistance

        cosine = CosineDistance()
        expected = (
            cosine.compute(mean_m, mean_c) + cosine.compute(skew_m, skew_c) + ProcrustesDistance().compute(cov_m, cov_c)
        )

        self.assertAlmostEqual(fitness._calculate_score(m_data, c_data), expected)

    def test_evaluate_scores_every_individual(self):
        # A larger, less degenerate dataset than setUp's: covariance/skewness
        # need enough rows to stay finite (see test_GAMIPY.py's TestRun for
        # the same, already-documented reasoning).
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "f1": rng.uniform(0, 10, 30),
                "f2": rng.uniform(0, 10, 30),
                "f3": rng.uniform(0, 10, 30),
            }
        )
        df.loc[3, "f1"] = None
        df.loc[7, "f2"] = None
        df.loc[15, "f3"] = None
        data = Data(df)
        population = RandomPopulation(data=data, population_size=5)
        population.initialize()
        m_data = Data(data.get_missing_data)
        c_data = Data(data.get_complete_data)

        fitness = Fitness(population=population, m_data=m_data, c_data=c_data)
        fitness.evaluate()
        for individual in population.population:
            self.assertTrue(individual.is_evaluated)
            self.assertTrue(np.isfinite(individual.score))

        # evaluate() should agree with an independent, direct
        # _calculate_score call on the same individual's imputed data -
        # proves evaluate() isn't just producing *a* score, but the
        # documented one.
        individual = population.population[0]
        imputed = fitness.impute_data(individual)
        m_data_ind = fitness.encoder.encode(Data(imputed))
        c_data_enc = fitness.encoder.encode(c_data)
        expected = fitness._calculate_score(m_data_ind, c_data_enc)
        self.assertAlmostEqual(individual.score, expected)


class TestUndefinedStatistics(unittest.TestCase):
    """Skewness needs >= 3 rows and covariance >= 2; an undefined statistic
    is left out of the score instead of making it NaN or crashing."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.c_data = Data(pd.DataFrame({"A": rng.uniform(0, 10, 20) ** 2, "B": rng.uniform(0, 10, 20)}))

    def test_one_row_m_data_scores_with_the_mean_only(self):
        m_data = Data(pd.DataFrame({"A": [3.0], "B": [4.0]}))
        fitness = Fitness(metric=CosineDistance())
        score = fitness._calculate_score(m_data, self.c_data)
        self.assertTrue(np.isfinite(score))
        mean_only = CosineDistance().compute(fitness.mean_score(m_data), fitness.mean_score(self.c_data))
        self.assertAlmostEqual(score, mean_only)

    def test_two_row_m_data_score_is_finite(self):
        m_data = Data(pd.DataFrame({"A": [3.0, 9.0], "B": [4.0, 1.0]}))
        self.assertTrue(np.isfinite(Fitness(metric=CosineDistance())._calculate_score(m_data, self.c_data)))
