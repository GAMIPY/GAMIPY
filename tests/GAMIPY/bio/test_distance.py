import unittest

from GAMIPY.bio.distance import (
    DISTANCE_METRICS,
    BrayCurtisDistance,
    CanberraDistance,
    ChebyshevDistance,
    CorrelationDistance,
    CosineDistance,
    DistanceMetric,
    EuclideanDistance,
    HammingDistance,
    JaccardDistance,
    ManhattanDistance,
    MinkowskiDistance,
    ProcrustesDistance,
)


class TestDistanceMetrics(unittest.TestCase):
    def test_cosine_distance_of_identical_vectors_is_zero(self):
        self.assertAlmostEqual(CosineDistance().compute([1, 2, 3], [1, 2, 3]), 0.0)

    def test_euclidean_distance(self):
        self.assertAlmostEqual(EuclideanDistance().compute([0, 0], [3, 4]), 5.0)

    def test_manhattan_distance(self):
        self.assertAlmostEqual(ManhattanDistance().compute([0, 0], [3, 4]), 7.0)

    def test_jaccard_distance_of_identical_vectors_is_zero(self):
        self.assertAlmostEqual(JaccardDistance().compute([1, 0, 1], [1, 0, 1]), 0.0)

    def test_hamming_distance(self):
        self.assertAlmostEqual(HammingDistance().compute([1, 0, 1, 1], [1, 1, 1, 0]), 0.5)

    def test_chebyshev_distance(self):
        self.assertAlmostEqual(ChebyshevDistance().compute([0, 0], [3, 4]), 4.0)

    def test_minkowski_distance_defaults_to_order_three(self):
        self.assertAlmostEqual(MinkowskiDistance().compute([0, 0], [1, 1]), (2) ** (1 / 3))

    def test_minkowski_distance_rejects_non_positive_order(self):
        with self.assertRaises(ValueError):
            MinkowskiDistance(p=0)

    def test_canberra_distance_of_identical_vectors_is_zero(self):
        self.assertAlmostEqual(CanberraDistance().compute([1, 2], [1, 2]), 0.0)

    def test_braycurtis_distance_of_identical_vectors_is_zero(self):
        self.assertAlmostEqual(BrayCurtisDistance().compute([1, 2], [1, 2]), 0.0)

    def test_correlation_distance_of_identical_vectors_is_zero(self):
        self.assertAlmostEqual(CorrelationDistance().compute([1, 2, 3], [1, 2, 3]), 0.0)

    def test_procrustes_distance_returns_zero_for_too_few_rows(self):
        self.assertEqual(ProcrustesDistance().compute([[1, 2]], [[1, 2]]), 0.0)

    def test_procrustes_distance_of_identical_matrices_is_zero(self):
        matrix = [[0, 0], [1, 1], [2, 0]]
        self.assertAlmostEqual(ProcrustesDistance().compute(matrix, matrix), 0.0)

    def test_distance_metrics_registry_is_instantiable(self):
        # Procrustes needs 2D input with at least two distinct rows; every
        # other registered metric accepts a plain vector.
        for name, metric_cls in DISTANCE_METRICS.items():
            with self.subTest(metric=name):
                if name == 'procrustes':
                    metric_cls().compute([[1, 2], [3, 4]], [[1, 2], [3, 4]])
                else:
                    metric_cls().compute([1, 2, 3], [1, 2, 3])

    def test_create_builds_the_named_metric(self):
        self.assertIsInstance(DistanceMetric.create('euclidean'), EuclideanDistance)
        self.assertIsInstance(DistanceMetric.create('cosine'), CosineDistance)

    def test_create_rejects_unknown_metric_name(self):
        with self.assertRaises(ValueError):
            DistanceMetric.create('not_a_real_metric')

    def test_create_rejects_procrustes(self):
        # Procrustes requires 2D input and is reserved for Fitness's fixed
        # covariance comparison, so it can't be selected as a general metric.
        with self.assertRaises(ValueError):
            DistanceMetric.create('procrustes')
