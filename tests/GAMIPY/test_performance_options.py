"""The GA's own accuracy and speed options (validation-based selection and
early stopping, standardized / full-data statistics, Gaussian mutation, elite
averaging, the vectorized fitness) and the supporting changes (shared
category codes, cached column profiles)."""
import random
import unittest

import numpy as np
import pandas as pd

from GAMIPY import GAMIPY
from GAMIPY.bio.encoding import CategoricalEncoder
from GAMIPY.bio.fitness import Fitness
from GAMIPY.bio.mutation import Mutation
from GAMIPY.config import GAConfig
from GAMIPY.data.categorical import CategoricalColumn
from GAMIPY.data.data import Data
from GAMIPY.data.numerical import NumericalColumn
from GAMIPY.population.gene import Gene
from GAMIPY.population.individual import Individual
from GAMIPY.runner import GeneticAlgorithmRunner


def numeric_frame(seed=0, rows=40):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=rows)
    df = pd.DataFrame({'a': a, 'b': 2 * a + rng.normal(scale=0.3, size=rows), 'c': rng.uniform(0, 5, rows)})
    return df.mask(rng.random(df.shape) < 0.15)


def seeded():
    random.seed(0)
    np.random.seed(0)


def runner_for(df, **options):
    config = {'population_size': 12, 'generations': 4, 'selection_number': 4, **options}
    return GeneticAlgorithmRunner(GAConfig(**config), Data(df))


class TestSharedCategoryCodes(unittest.TestCase):

    def test_same_label_gets_the_same_code_in_every_dataset(self):
        full = Data(pd.DataFrame({'c': ['x', 'y', 'z', 'x']}))
        encoder = CategoricalEncoder.for_data(full)
        first = encoder.encode(Data(pd.DataFrame({'c': ['z', 'x']}))).data['c'].tolist()
        second = encoder.encode(Data(pd.DataFrame({'c': ['x', 'z']}))).data['c'].tolist()
        self.assertEqual(first, [2, 0])
        self.assertEqual(second, [0, 2])

    def test_without_classes_each_dataset_uses_its_own_order(self):
        # The old behaviour, still the default when no classes are given.
        encoder = CategoricalEncoder()
        self.assertEqual(encoder.encode(Data(pd.DataFrame({'c': ['z', 'x']}))).data['c'].tolist(), [0, 1])


class TestColumnProfile(unittest.TestCase):

    def test_profile_matches_the_column_classes(self):
        df = pd.DataFrame({'n': [3.0, None, 1.0, 3.0], 'c': ['b', None, 'a', 'b']})
        data = Data(df)
        num, cat = NumericalColumn(data, 'n'), CategoricalColumn(data, 'c')
        profile = data.column_profile('n')
        self.assertEqual(
            (profile.minimum, profile.maximum, profile.observed, profile.distinct),
            (num.min_val, num.max_val, 3, 2),
        )
        cat_profile = data.column_profile('c')
        self.assertEqual(cat_profile.classes, cat.classes)
        self.assertEqual(cat_profile.mode, cat.mode)

    def test_cache_follows_the_dataframe(self):
        data = Data(pd.DataFrame({'n': [1.0, 2.0]}))
        self.assertEqual(data.column_profile('n').maximum, 2.0)
        data.data = pd.DataFrame({'n': [5.0, 9.0]})
        self.assertEqual(data.column_profile('n').maximum, 9.0)
        self.assertEqual(data.get_numeric_columns, ['n'])


class TestFitnessOptions(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        self.frame = pd.DataFrame({'a': rng.normal(size=30), 'b': rng.normal(size=30) ** 2})

    def test_standardized_score_is_zero_for_identical_data(self):
        fitness = Fitness(standardize=True)
        summary = fitness._summary(self.frame)
        self.assertAlmostEqual(fitness._standardized_score(self.frame, summary), 0.0)

    def test_standardized_score_is_scale_free(self):
        # Rescaling both datasets (e.g. a column in grams vs kilograms) must
        # not change the score: z-scored means, skewness and correlations are
        # all invariant to positive scaling.
        fitness = Fitness(standardize=True)
        shifted = self.frame + 1.0
        a = fitness._standardized_score(shifted, fitness._summary(self.frame))
        b = fitness._standardized_score(shifted * 1000, fitness._summary(self.frame * 1000))
        self.assertGreater(a, 0)
        self.assertAlmostEqual(a, b)


class TestValidation(unittest.TestCase):

    def test_validation_cells_are_hidden_and_restored(self):
        df = numeric_frame()
        seeded()
        runner = runner_for(df, generations=3, validation_fraction=0.1)
        observed = int(df.notna().to_numpy().sum())
        self.assertEqual(len(runner.validation), round(0.1 * observed))
        self.assertGreater(int(runner.data.data.isna().to_numpy().sum()), int(df.isna().to_numpy().sum()))
        best = runner.run()
        result = runner.build_result(best)
        self.assertFalse(result.isna().any().any())
        observed_mask = df.notna()
        pd.testing.assert_frame_equal(result[observed_mask], df[observed_mask])
        self.assertTrue(np.isfinite(runner.best_validation_error))

    def test_patience_stops_early(self):
        seeded()
        runner = runner_for(numeric_frame(), generations=60, validation_fraction=0.1, patience=2)
        runner.run()
        self.assertLess(runner.generations_run, 60)

    def test_without_validation_nothing_changes(self):
        runner = GeneticAlgorithmRunner(GAConfig(validation_fraction=0.0), Data(numeric_frame()))
        self.assertEqual(runner.validation, {})
        self.assertIs(runner.data, runner.original_data)


class TestGaussianMutation(unittest.TestCase):

    def test_values_stay_in_the_observed_range_and_near_the_original(self):
        df = numeric_frame()
        data = Data(df)
        mutation = Mutation(mutation_rate=1.0, data=data, mutation_type='gaussian', mutation_scale=0.1)
        ind = Individual(0, missing_indices=[(0, 0, 'c')], data=data)
        ind.genes = [Gene(orig_location=(0, 'c'), m_location=(0, 'c'), name='c', data=2.5, dtype='num')]
        np.random.seed(0)
        values = [mutation.mutate(ind).genes[0].data for _ in range(200)]
        low, high = df['c'].min(), df['c'].max()
        self.assertTrue(all(low <= v <= high for v in values))
        self.assertLess(abs(np.mean(values) - 2.5), 0.2)
        self.assertLess(np.std(values), 0.5 * df['c'].std())

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            Mutation(mutation_type='sideways')


class TestDefaultIsAPureGA(unittest.TestCase):

    def test_imputes_every_missing_cell_and_keeps_observed_ones(self):
        rng = np.random.default_rng(4)
        df = pd.DataFrame({'a': rng.normal(size=40), 'b': rng.normal(size=40),
                           'c': rng.choice(['x', 'y', 'z'], 40)})
        df = df.mask(rng.random(df.shape) < 0.15)
        seeded()
        gamipy = GAMIPY(data=df, population_size=15, generations=6, selection_number=4, patience=3)
        gamipy.run()
        result = gamipy.imputed_data
        self.assertFalse(result.isna().any().any())
        observed = df.notna()
        pd.testing.assert_frame_equal(result[observed], df[observed])
        self.assertTrue(set(result['c']) <= {'x', 'y', 'z'})

    def test_default_initialization_is_random(self):
        self.assertEqual(GAConfig().initialization_strategy, 'random')


class TestEliteMean(unittest.TestCase):

    def test_averages_numbers_and_votes_on_categories(self):
        runner = runner_for(numeric_frame(), validation_fraction=0.0)
        elite = []
        for value, label in [(1.0, 'x'), (2.0, 'y'), (6.0, 'y')]:
            ind = Individual(0, missing_indices=[(0, 0, 'n')], data=runner.m_data)
            ind.genes = [Gene(orig_location=(0, 'n'), m_location=(0, 'n'), name='n', data=value, dtype='num'),
                         Gene(orig_location=(0, 'c'), m_location=(0, 'c'), name='c', data=label, dtype='cat')]
            elite.append(ind)
        mean = runner._elite_mean(elite)
        self.assertEqual([g.data for g in mean.genes], [3.0, 'y'])

    def test_run_returns_a_complete_imputation(self):
        df = numeric_frame()
        seeded()
        runner = runner_for(df, result_method='elite_mean', elite_size=3)
        result = runner.build_result(runner.run())
        self.assertFalse(result.isna().any().any())
        observed = df.notna()
        pd.testing.assert_frame_equal(result[observed], df[observed])

    def test_unknown_result_method_raises(self):
        with self.assertRaises(ValueError):
            runner_for(numeric_frame(), result_method='median').run()


class TestMoments(unittest.TestCase):
    """GAMIPY.bio.moments must reproduce pandas' statistics."""

    def test_match_pandas_with_a_shared_prefix(self):
        from GAMIPY.bio.moments import Moments
        rng = np.random.default_rng(0)
        scale, offset = np.array([1.0, 10.0, 1000.0, 0.0]), np.array([0.0, 5.0, 1e4, 2.5])
        prefix = rng.normal(size=(7, 4)) * scale + offset  # column 3 is constant
        own = rng.normal(size=(3, 25, 4)) * scale + offset
        combined = Moments.of(prefix.T[None]).combine(Moments.of(own.transpose(0, 2, 1)))
        for i in range(3):
            frame = pd.DataFrame(np.vstack([prefix, own[i]]))
            np.testing.assert_allclose(combined.mean[i], frame.mean(), rtol=1e-12)
            np.testing.assert_allclose(combined.skew()[i], frame.skew(), rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(combined.cov()[i], frame.cov(), rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(combined.corr()[i], frame.corr(), rtol=1e-9, atol=1e-9, equal_nan=True)

    def test_too_few_rows_for_skewness(self):
        from GAMIPY.bio.moments import Moments
        self.assertTrue(np.isnan(Moments.of(np.ones((1, 2, 2))).skew()).all())


class TestVectorizedFitness(unittest.TestCase):
    """Fitness.evaluate scores a population in NumPy; the scores must match
    the pandas computation (one DataFrame per individual) they replace."""

    @staticmethod
    def mixed_frame(seed=3, rows=60):
        rng = np.random.default_rng(seed)
        a = rng.normal(size=rows)
        df = pd.DataFrame({
            'a': a * 1000 + 5000,  # a large offset, where naive moment formulas lose precision
            'b': 2 * a + rng.normal(scale=0.5, size=rows),
            'n': rng.integers(0, 5, rows).astype(float),
            'c': rng.choice(['x', 'y', 'z'], rows),
        })
        return df.mask(rng.random(df.shape) < 0.12)

    @staticmethod
    def pandas_scores(fitness):
        encoded_c = fitness.encoder.encode(fitness.c_data)
        reference = fitness.reference if fitness.reference is not None else encoded_c
        summary = fitness._summary(reference.data) if fitness.standardize else None
        scores = []
        for individual in fitness.population.population:
            m_data = fitness.encoder.encode(Data(fitness.impute_data(individual)))
            if fitness.full_data_statistics:
                m_data = Data(pd.concat([encoded_c.data, m_data.data], ignore_index=True))
            scores.append(fitness._standardized_score(m_data.data, summary) if summary is not None
                          else fitness._calculate_score(m_data, reference))
        return scores

    def fitness_for(self, **options):
        seeded()
        config = GAConfig(population_size=12, validation_fraction=0.0, **options)
        runner = GeneticAlgorithmRunner(config, Data(self.mixed_frame()))
        population = runner.initialize_population()
        return runner.factory.fitness(population, runner.m_data, runner.c_data, **runner.fitness_context())

    def assert_matches_pandas(self, fitness):
        expected = self.pandas_scores(fitness)
        fitness.evaluate()
        for individual, score in zip(fitness.population.population, expected, strict=True):
            self.assertAlmostEqual(individual.score, score, delta=1e-9 * max(1.0, abs(score)))

    def test_default_options(self):
        self.assert_matches_pandas(self.fitness_for())

    def test_every_combination_of_statistics_options(self):
        for standardize in (True, False):
            for full in (True, False):
                with self.subTest(standardize=standardize, full=full):
                    self.assert_matches_pandas(self.fitness_for(standardize=standardize, full_data_statistics=full))

    def test_every_complete_individual_takes_the_vectorized_path(self):
        # With a categorical column, the first individual of a stack once fell
        # back to pandas (same score, much slower): its labels were encoded
        # with the previous individual's gene layout.
        from GAMIPY.bio.fitness import _ImputedRows
        fitness = self.fitness_for()
        encoded_c = fitness.encoder.encode(fitness.c_data)
        rows = _ImputedRows.build(fitness.m_data, fitness.encoder, encoded_c.data)
        _, complete = rows.stack(fitness.population.population)
        self.assertTrue(complete.all())

    def test_an_individual_with_an_empty_cell_is_scored_with_pandas(self):
        fitness = self.fitness_for()
        fitness.population.population[0].genes.pop()
        self.assert_matches_pandas(fitness)

    def test_without_fixed_codes_everything_is_scored_with_pandas(self):
        fitness = self.fitness_for()
        fitness.encoder = CategoricalEncoder()
        self.assert_matches_pandas(fitness)


if __name__ == '__main__':
    unittest.main()
