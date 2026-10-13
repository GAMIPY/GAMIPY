import random
import unittest

import numpy as np
import pandas as pd

from GAMIPY.bio.distance import CosineDistance, EuclideanDistance
from GAMIPY.bio.selection import Elitist, Rank, SteadyState, Tournament
from GAMIPY.datasets.sample_data import SampleData
from GAMIPY.GAMIPY import GAMIPY
from GAMIPY.population.population import (
    ConstantPopulation,
    HeuristicPopulation,
    HybridPopulation,
    OnesPopulation,
    RandomPopulation,
    ZerosPopulation,
)


class TestInitializePopulation(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'A': [None, 2, 3],
            'B': [4, 5, 6],
            'C': ["No", "Yes", None]
        })
        self.population_size = 10

    def _make(self, strategy: str, **kwargs) -> GAMIPY:
        return GAMIPY(data=self.df, population_size=self.population_size, initialization_strategy=strategy, **kwargs)

    def test_random_strategy_uses_default_population_size(self):
        gamipy = self._make('random')
        gamipy.initialize_population()
        self.assertIsInstance(gamipy.population, RandomPopulation)
        self.assertEqual(len(gamipy.population.population), self.population_size)

    def test_zeros_strategy_uses_default_population_size(self):
        gamipy = self._make('zeros')
        gamipy.initialize_population()
        self.assertIsInstance(gamipy.population, ZerosPopulation)
        self.assertEqual(len(gamipy.population.population), self.population_size)

    def test_ones_strategy_uses_default_population_size(self):
        gamipy = self._make('ones')
        gamipy.initialize_population()
        self.assertIsInstance(gamipy.population, OnesPopulation)
        self.assertEqual(len(gamipy.population.population), self.population_size)

    def test_constant_strategy_uses_default_population_size(self):
        gamipy = self._make('constant')
        gamipy.initialize_population()
        self.assertIsInstance(gamipy.population, ConstantPopulation)
        self.assertEqual(len(gamipy.population.population), self.population_size)

    def test_explicit_pop_size_overrides_default(self):
        gamipy = self._make('random')
        gamipy.initialize_population(pop_size=3)
        self.assertEqual(len(gamipy.population.population), 3)

    def test_gene_orig_location_is_a_true_original_row_not_m_data_relative(self):
        # GAMIPY.__init__ now computes self.missing_indices from the true
        # original Data (not from m_data, which is row-filtered and
        # reset-index'd) and passes it down to Population/Individual, so
        # Gene.location (orig_location) is a real row label in the user's
        # original DataFrame. A non-default, non-contiguous index makes this
        # a meaningful check: before this fix, gene.location[0] would have
        # been an m_data-relative 0..N-1 position instead.
        df = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'f2': [6.0, 7.0, 8.0, 9.0, 10.0],
        }, index=[10, 11, 13, 14, 15])
        df.loc[11, 'f1'] = None
        df.loc[14, 'f2'] = None
        true_missing_rows = set(df.index[df.isnull().any(axis=1)])

        gamipy = GAMIPY(data=df, population_size=3, initialization_strategy='random', validation_fraction=0.0)
        gamipy.initialize_population()

        for individual in gamipy.population.population:
            for gene in individual.genes:
                self.assertIn(gene.location[0], true_missing_rows)

    def test_heuristic_strategy_without_heuristic_population_raises_value_error(self):
        # GAMIPY.__init__ now defines self.heuristic_population (default None),
        # so this is a clear, expected ValueError from HeuristicPopulation.initialize()
        # (missing required data), not the old AttributeError from the attribute
        # never existing at all.
        gamipy = self._make('heuristic')
        with self.assertRaises(ValueError):
            gamipy.initialize_population()

    def test_heuristic_strategy_with_heuristic_population_succeeds(self):
        heuristic_population = pd.DataFrame({0: [0, 2], 1: [1, 2], 2: ['A', 'C'], 3: [10, 'Maybe']})
        gamipy = self._make('heuristic', heuristic_population=heuristic_population)
        gamipy.initialize_population()
        self.assertIsInstance(gamipy.population, HeuristicPopulation)
        self.assertEqual(len(gamipy.population.population), self.population_size)

    def test_hybrid_strategy_without_heuristic_population_raises_value_error(self):
        # Hybrid builds its seeded half through HeuristicPopulation, so a
        # missing heuristic_population now gets the same clear ValueError
        # (it used to be an accidental AttributeError on None.shape).
        gamipy = self._make('hybrid')
        with self.assertRaises(ValueError):
            gamipy.initialize_population()

    def test_hybrid_strategy_with_heuristic_population_succeeds(self):
        heuristic_population = pd.DataFrame({0: [0, 2], 1: [1, 2], 2: ['A', 'C'], 3: [10, 'Maybe']})
        gamipy = self._make('hybrid', heuristic_population=heuristic_population)
        gamipy.initialize_population()
        self.assertIsInstance(gamipy.population, HybridPopulation)
        self.assertEqual(len(gamipy.population.population), self.population_size)

    def test_invalid_strategy_raises_value_error(self):
        gamipy = self._make('not_a_real_strategy')
        with self.assertRaises(ValueError):
            gamipy.initialize_population()


class TestSelectionProperty(unittest.TestCase):
    """Covers GAMIPY.selection (previously untested)."""

    def setUp(self):
        # Each numeric column with a missing value must still have at least
        # one non-null value in the missing-rows subset GAMIPY builds
        # (self.m_data), otherwise NumericalColumn.min_val/max_val come back
        # NaN and np.random.uniform(nan, nan) raises OverflowError.
        self.df = pd.DataFrame({
            'feature1': [1, 2, None],
            'feature2': [4, None, 6],
        })
        self.population_size = 20
        self.selection_number = 5

    def _make(self, selection_method: str, **kwargs) -> GAMIPY:
        gamipy = GAMIPY(
            data=self.df,
            population_size=self.population_size,
            selection_method=selection_method,
            selection_number=self.selection_number,
            tournament_size=kwargs.pop('tournament_size', 3),
            **kwargs,
        )
        gamipy.initialize_population()
        return gamipy

    def test_elitist_selection_returns_selection_number_individuals(self):
        gamipy = self._make('elitist')
        self.assertIsInstance(gamipy.selection, Elitist)
        selected = gamipy.selection.select()
        self.assertEqual(len(selected), self.selection_number)

    def test_tournament_selection_returns_selection_number_individuals(self):
        gamipy = self._make('tournament')
        self.assertIsInstance(gamipy.selection, Tournament)
        selected = gamipy.selection.select()
        self.assertEqual(len(selected), self.selection_number)

    def test_rank_selection_returns_selection_number_individuals(self):
        gamipy = self._make('rank')
        self.assertIsInstance(gamipy.selection, Rank)
        selected = gamipy.selection.select()
        self.assertEqual(len(selected), self.selection_number)

    def test_roulette_selection_on_unevaluated_population_raises_value_error(self):
        # Unevaluated individuals (Individual.UNEVALUATED_SCORE) are left out
        # of the draw instead of being summed into an overflowing total, so
        # with none evaluated this is a clear ValueError rather than the old
        # silent [].
        gamipy = self._make('roulette')
        with self.assertRaises(ValueError):
            gamipy.selection.select()

    def test_roulette_selection_returns_selection_number_individuals(self):
        gamipy = self._make('roulette')
        for i, ind in enumerate(gamipy.population.population):
            ind.set_score(float(i + 1))
        selected = gamipy.selection.select()
        self.assertEqual(len(selected), self.selection_number)

    def test_stochastic_universal_selection_on_unevaluated_population_raises_value_error(self):
        # Unevaluated individuals are left out of the draw; with none
        # evaluated SUS raises instead of returning [].
        gamipy = self._make('stochastic_universal')
        with self.assertRaises(ValueError):
            gamipy.selection.select()

    def test_stochastic_universal_selection_returns_selection_number_individuals(self):
        gamipy = self._make('stochastic_universal')
        for i, ind in enumerate(gamipy.population.population):
            ind.set_score(float(i + 1))
        selected = gamipy.selection.select()
        self.assertEqual(len(selected), self.selection_number)

    def test_steady_state_selection_uses_ss_number(self):
        gamipy = self._make('steady_state', ss_number=2)
        selection = gamipy.selection
        self.assertIsInstance(selection, SteadyState)
        self.assertEqual(selection.ss_number, 2)
        # SteadyState.select() drops ss_number individuals rather than
        # returning selection_number of them (selection.py's own TODO).
        selected = selection.select()
        self.assertEqual(len(selected), self.population_size - 2)

    def test_steady_state_ss_number_defaults_to_one(self):
        gamipy = self._make('steady_state')
        self.assertEqual(gamipy.selection.ss_number, 1)

    def test_selection_options_reach_their_strategy(self):
        gamipy = self._make('tournament', tournament_size=4)
        self.assertEqual(gamipy.selection.tournament_size, 4)
        gamipy = self._make('steady_state', ss_number=3)
        self.assertEqual(gamipy.selection.ss_number, 3)

    def test_invalid_selection_method_raises_value_error(self):
        gamipy = self._make('not_a_real_method')
        with self.assertRaises(ValueError):
            _ = gamipy.selection


class TestFitnessMetricProperty(unittest.TestCase):
    """
    Covers GAMIPY.fitness_metric (fitness_method used to
    be accepted and stored but never actually used - Fitness._calculate_score
    hardcoded CosineDistance/ProcrustesDistance with no way to select any of
    the ~9 other registered distance metrics).
    """

    def setUp(self):
        self.df = pd.DataFrame({'A': [1.0, None, 3.0], 'B': [4.0, 5.0, 6.0]})

    def test_default_fitness_method_resolves_to_cosine(self):
        gamipy = GAMIPY(data=self.df)
        self.assertIsInstance(gamipy.fitness_metric, CosineDistance)

    def test_fitness_method_selects_the_matching_metric(self):
        gamipy = GAMIPY(data=self.df, fitness_method='euclidean')
        self.assertIsInstance(gamipy.fitness_metric, EuclideanDistance)

    def test_invalid_fitness_method_raises_value_error(self):
        gamipy = GAMIPY(data=self.df, fitness_method='not_a_real_metric')
        with self.assertRaises(ValueError):
            _ = gamipy.fitness_metric

    def test_procrustes_is_not_a_selectable_fitness_method(self):
        # Reserved internally, fixed, for the covariance comparison - it
        # requires 2D input and can't serve as a general vector metric.
        gamipy = GAMIPY(data=self.df, fitness_method='procrustes')
        with self.assertRaises(ValueError):
            _ = gamipy.fitness_metric

    def test_fitness_method_actually_changes_the_computed_score(self):
        # Proves fitness_method is genuinely wired into scoring, not just
        # accepted and ignored: Fitness._calculate_score on the same two
        # datasets should give a different score depending on which metric
        # it's built with.
        from GAMIPY.bio.fitness import Fitness
        from GAMIPY.data.data import Data

        # Skewed (non-symmetric) columns, so the skewness vectors aren't
        # all-zero - an all-zero skewness vector would make cosine distance
        # divide by zero (a separate, already-documented NaN edge case, not
        # what this test is about).
        rng = np.random.default_rng(0)
        m_data = Data(pd.DataFrame({'A': rng.uniform(0, 10, 10), 'B': rng.uniform(0, 10, 10)}))
        c_data = Data(pd.DataFrame({'A': rng.uniform(0, 10, 10) ** 2, 'B': rng.uniform(0, 10, 10)}))

        cosine_score = Fitness(metric=CosineDistance())._calculate_score(m_data, c_data)
        euclidean_score = Fitness(metric=EuclideanDistance())._calculate_score(m_data, c_data)

        self.assertTrue(np.isfinite(cosine_score) and np.isfinite(euclidean_score))
        self.assertNotAlmostEqual(cosine_score, euclidean_score)


class TestCrossoverProperty(unittest.TestCase):
    """Covers GAMIPY.crossover (previously untested)."""

    def setUp(self):
        # All-numeric, >=3 missing values so every individual has >=3 genes,
        # enough for two-point crossover's two cut points.
        self.df = pd.DataFrame({
            'f1': [1.0, None, 3.0, 4.0],
            'f2': [5.0, 6.0, None, 8.0],
            'f3': [9.0, 10.0, 11.0, None],
        })
        self.population_size = 5

    def _parents(self, crossover_type: str):
        gamipy = GAMIPY(
            data=self.df,
            population_size=self.population_size,
            crossover_type=crossover_type,
            initialization_strategy='random',
        )
        gamipy.initialize_population()
        parent1, parent2 = gamipy.population.population[0], gamipy.population.population[1]
        return gamipy, parent1, parent2

    def test_uniform_crossover_returns_two_children(self):
        gamipy, parent1, parent2 = self._parents('uniform')
        child1, child2 = gamipy.crossover.run(parent1, parent2)
        self.assertEqual(len(child1.genes), len(parent1.genes))
        self.assertEqual(len(child2.genes), len(parent2.genes))

    def test_one_point_crossover_returns_two_children(self):
        gamipy, parent1, parent2 = self._parents('one_point')
        child1, child2 = gamipy.crossover.run(parent1, parent2)
        self.assertEqual(len(child1.genes), len(parent1.genes))
        self.assertEqual(len(child2.genes), len(parent2.genes))

    def test_two_point_crossover_returns_two_children(self):
        # Previously zero coverage (TestTwoPoint was entirely commented out
        # in tests/GAMIPY/bio/test_crossover.py).
        gamipy, parent1, parent2 = self._parents('two_point')
        child1, child2 = gamipy.crossover.run(parent1, parent2)
        self.assertEqual(len(child1.genes), len(parent1.genes))
        self.assertEqual(len(child2.genes), len(parent2.genes))

    def test_arithmetic_crossover_returns_one_gene_per_missing_value(self):
        gamipy, parent1, parent2 = self._parents('arithmetic')
        child1, child2 = gamipy.crossover.run(parent1, parent2)
        self.assertEqual(len(child1.genes), len(parent1.genes))
        self.assertEqual(len(child2.genes), len(parent2.genes))
        for gene1, gene2, child_gene in zip(parent1.genes, parent2.genes, child1.genes, strict=True):
            self.assertEqual(child_gene._m_location, gene1._m_location)
            self.assertIsInstance(child_gene.data, float)
            self.assertAlmostEqual(child_gene.data, (gene1.data + gene2.data) / 2)

    def test_blend_crossover_returns_one_gene_per_missing_value(self):
        gamipy, parent1, parent2 = self._parents('blend')
        child1, child2 = gamipy.crossover.run(parent1, parent2)
        self.assertEqual(len(child1.genes), len(parent1.genes))
        self.assertEqual(len(child2.genes), len(parent2.genes))
        genes = zip(parent1.genes, parent2.genes, child1.genes, child2.genes, strict=True)
        for gene1, gene2, c1_gene, c2_gene in genes:
            self.assertEqual(c1_gene._m_location, gene1._m_location)
            self.assertIsInstance(c1_gene.data, float)
            # alpha-independent invariant, see TestBlend.test_crossover in
            # tests/GAMIPY/bio/test_crossover.py.
            self.assertAlmostEqual(c1_gene.data + c2_gene.data, gene1.data + gene2.data)

    def test_invalid_crossover_type_raises_value_error(self):
        # Used to silently return None (no else/raise in the property), so
        # the failure only surfaced later as an AttributeError in run().
        gamipy, _, _ = self._parents("not_a_real_type")
        with self.assertRaises(ValueError):
            _ = gamipy.crossover

    def test_crossover_property_passes_a_list_not_a_population_object(self):
        # GAMIPY.crossover used to pass self.population
        # (a Population object) instead of self.population.population (a
        # list of Individuals) to every Crossover subclass - harmless while
        # nothing read it, but Crossover.is_all_numerical() does
        # `for ind in self.population`, which raises TypeError on a
        # Population object (not iterable) instead of a list.
        gamipy, _, _ = self._parents("arithmetic")
        self.assertIsInstance(gamipy.crossover.population, list)
        self.assertTrue(gamipy.crossover.is_all_numerical())


class TestRun(unittest.TestCase):
    """Covers GAMIPY.run() (previously untested)."""

    def setUp(self):
        # A larger, less degenerate dataset than the other test classes use:
        # Fitness.evaluate() (called every generation inside run()) computes
        # covariance/skewness/Procrustes distance between imputed and
        # original data, which needs enough rows to stay finite. A tiny
        # 3-4 row frame trips an unrelated, already-documented bug
        # ("Covariance and skewness returns nan for some cases") and isn't
        # what these tests are checking.
        rng = np.random.default_rng(0)
        self.df = pd.DataFrame(
            {
                "f1": rng.uniform(0, 10, 30),
                "f2": rng.uniform(0, 10, 30),
                "f3": rng.uniform(0, 10, 30),
            }
        )
        self.df.loc[3, "f1"] = None
        self.df.loc[7, "f2"] = None
        self.df.loc[15, "f3"] = None

    def test_run_completes_and_imputed_data_is_a_complete_dataframe(self):
        gamipy = GAMIPY(representation='cells',
            data=self.df,
            population_size=20,
            generations=2,
            selection_number=5,
            tournament_size=3,
        )
        gamipy.run()
        self.assertIsInstance(gamipy.imputed_data, pd.DataFrame)
        self.assertEqual(gamipy.imputed_data.shape, self.df.shape)
        self.assertEqual(list(gamipy.imputed_data.index), list(self.df.index))
        self.assertFalse(gamipy.imputed_data.isnull().any().any())
        # Originally-complete rows are untouched by imputation.
        complete_rows = self.df.dropna().index
        pd.testing.assert_frame_equal(gamipy.imputed_data.loc[complete_rows], self.df.loc[complete_rows])

    def test_run_imputes_categorical_column_with_real_labels(self):
        # A categorical missing cell must be filled with one of the column's
        # real labels, not CategoricalCodec.encode's integer code -
        # RandomIndividual.individual() used to write the encoded int
        # straight into gene.data.
        rng = np.random.default_rng(0)
        df = self.df.copy()
        df["cat"] = rng.choice(["red", "green", "blue"], len(df))
        df.loc[20, "cat"] = None

        gamipy = GAMIPY(representation='cells',
            data=df,
            population_size=20,
            generations=2,
            selection_number=5,
            tournament_size=3,
        )
        gamipy.run()
        self.assertIn(gamipy.imputed_data.loc[20, "cat"], ["red", "green", "blue"])

    def test_run_with_selection_number_one_raises_value_error(self):
        # Crossover needs two parents; run() now checks len(selected) and
        # raises a clear ValueError instead of an IndexError on selected[1].
        gamipy = GAMIPY(representation='cells', data=self.df, population_size=20, generations=2, selection_number=1)
        with self.assertRaises(ValueError):
            gamipy.run()

    def test_every_selection_method_honours_select_contract_in_run(self):
        # LSP: run() must work whichever Selection subclass is plugged in.
        for method in ["elitist", "tournament", "roulette", "rank", "steady_state", "stochastic_universal"]:
            with self.subTest(method=method):
                gamipy = GAMIPY(representation='cells',
                    data=self.df,
                    population_size=20,
                    generations=3,
                    selection_number=5,
                    tournament_size=3,
                    selection_method=method,
                )
                gamipy.run()
                self.assertFalse(gamipy.imputed_data.isnull().any().any())
                self.assertEqual(len(gamipy.population.population), 20)

    def test_run_with_invalid_crossover_type_raises_value_error(self):
        # Fails up front, before the first generation, rather than as an
        # AttributeError on `None.run(...)` inside the loop.
        gamipy = GAMIPY(representation='cells',
            data=self.df,
            population_size=20,
            generations=2,
            selection_number=5,
            crossover_type="not_a_real_type",
        )
        with self.assertRaises(ValueError):
            gamipy.run()

    def test_run_with_invalid_selection_method_raises_value_error(self):
        gamipy = GAMIPY(representation='cells',
            data=self.df,
            population_size=20,
            generations=2,
            selection_number=5,
            selection_method="not_a_real_method",
        )
        with self.assertRaises(ValueError):
            gamipy.run()

    def test_run_with_invalid_initialization_strategy_raises_value_error(self):
        gamipy = GAMIPY(representation='cells',
            data=self.df,
            population_size=20,
            generations=2,
            selection_number=5,
            initialization_strategy="not_a_real_strategy",
        )
        with self.assertRaises(ValueError):
            gamipy.run()

    def test_run_with_steady_state_selection_completes(self):
        gamipy = GAMIPY(representation='cells',
            data=self.df,
            population_size=20,
            generations=2,
            selection_number=5,
            selection_method="steady_state",
            ss_number=2,
        )
        gamipy.run()
        self.assertIsInstance(gamipy.imputed_data, pd.DataFrame)
        self.assertFalse(gamipy.imputed_data.isnull().any().any())

    def test_run_keeps_population_size_fixed_across_generations(self):
        # SteadyState.select() returns population_size - ss_number individuals,
        # not selection_number; run() used to size the refill from
        # selection_number, so the population grew every generation
        # (20 -> 33 -> ...).
        for method, extra in [("steady_state", {"ss_number": 1}), ("steady_state", {"ss_number": 2}), ("elitist", {})]:
            with self.subTest(method=method, **extra):
                gamipy = GAMIPY(representation='cells',
                    data=self.df,
                    population_size=20,
                    generations=4,
                    selection_number=5,
                    selection_method=method,
                    **extra,
                )
                gamipy.run()
                self.assertEqual(len(gamipy.population.population), 20)


class TestRandomState(unittest.TestCase):
    """Covers ``random_state``: one seed makes run() reproducible in both
    representations (the cells GA draws from NumPy's and Python's random
    number generators) and leaves the caller's generators as they were."""

    def setUp(self):
        iris = SampleData().load_data("iris")
        self.damaged = iris.mask(np.random.default_rng(1).random(iris.shape) < 0.2)

    def _run(self, representation: str, random_state: int | None) -> pd.DataFrame:
        gamipy = GAMIPY(data=self.damaged, representation=representation, population_size=20,
                        generations=5, random_state=random_state)
        gamipy.run()
        assert gamipy.imputed_data is not None
        return gamipy.imputed_data

    def test_same_seed_gives_the_same_imputation(self):
        for representation in ('model', 'cells'):
            with self.subTest(representation=representation):
                # Disturb both global generators between the runs: the seed alone must decide.
                first = self._run(representation, 7)
                np.random.random()
                random.random()
                pd.testing.assert_frame_equal(first, self._run(representation, 7))

    def test_different_seeds_give_different_imputations(self):
        for representation in ('model', 'cells'):
            with self.subTest(representation=representation):
                self.assertFalse(self._run(representation, 7).equals(self._run(representation, 8)))

    def test_run_restores_the_callers_generator_states(self):
        np.random.seed(123)
        random.seed(123)
        numpy_before, python_before = np.random.random(), random.random()
        np.random.seed(123)
        random.seed(123)
        self._run('cells', 7)
        self.assertEqual(np.random.random(), numpy_before)
        self.assertEqual(random.random(), python_before)

    def test_seed_set_on_config_after_construction_is_used(self):
        gamipy = GAMIPY(data=self.damaged, population_size=20, generations=5)
        gamipy.config.random_state = 7
        gamipy.run()
        pd.testing.assert_frame_equal(gamipy.imputed_data, self._run('model', 7))


class TestImputeData(unittest.TestCase):
    """Covers GAMIPY.impute_data (writes an individual's genes back onto a
    copy of the original full dataset)."""

    def test_maps_genes_back_to_correct_original_row_with_non_default_index(self):
        # A gene's `_m_location[0]` is a position within the missing-only,
        # reset-index m_data, not a label into the original data - this
        # test uses a non-contiguous, non-zero-based original index so a
        # naive `.loc[gene._m_location[0], ...]` would raise KeyError (or,
        # on a default RangeIndex, could silently write to the wrong row).
        df = pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "f2": [6.0, 7.0, 8.0, 9.0, 10.0],
            },
            index=[10, 11, 13, 14, 15],
        )
        df.loc[11, "f1"] = None
        df.loc[14, "f2"] = None

        gamipy = GAMIPY(data=df, population_size=5, initialization_strategy="random")
        gamipy.initialize_population()
        individual = gamipy.population.population[0]

        imputed = gamipy.impute_data(individual)

        self.assertIsInstance(imputed, pd.DataFrame)
        self.assertEqual(list(imputed.index), list(df.index))
        self.assertFalse(imputed.isnull().any().any())
        # Untouched original values for cells that were never missing.
        self.assertEqual(imputed.loc[10, "f1"], 1.0)
        self.assertEqual(imputed.loc[11, "f2"], 7.0)
        self.assertEqual(imputed.loc[14, "f1"], 4.0)


if __name__ == "__main__":
    unittest.main()
