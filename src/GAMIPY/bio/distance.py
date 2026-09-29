from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import scipy.spatial as sp


class DistanceMetric(ABC):

    """
    A single distance metric strategy, comparing two same-shaped vectors or
    matrices. Split out of Fitness, which used to define all ~11 of these as
    bound methods on itself alongside two unrelated responsibilities
    (categorical encoding and imputed-data construction). Each metric is now its own class
    behind a single `compute` method, so Fitness (or anything scoring two
    datasets) can depend on just the one metric it needs instead of the
    whole ~15-method Fitness interface.
    """

    @staticmethod
    def create(method: str) -> "DistanceMetric":

        """
        Builds the distance metric registered under `method` in DISTANCE_METRICS.

        Args
        ----
        - **method (str)**: A key of DISTANCE_METRICS, e.g. 'cosine' or 'euclidean'.

        Raises
        ------
        - **ValueError**: If `method` is not a registered metric, or is
          'procrustes'. Procrustes analysis requires two-dimensional input
          (it compares whole covariance matrices after optimal scaling/
          rotation, not simple vectors), so unlike the other ten metrics it
          can't stand in for a general-purpose vector metric - `Fitness`
          always uses `ProcrustesDistance` directly, fixed, for the
          covariance comparison regardless of this choice.
        """

        selectable = {name: cls for name, cls in DISTANCE_METRICS.items() if name != 'procrustes'}
        try:
            metric_cls = selectable[method]
        except KeyError:
            raise ValueError(
                "Invalid distance metric. Choose from "
                f"{', '.join(sorted(selectable))}."
            ) from None
        return metric_cls()

    @abstractmethod
    def compute(self, matrix1: npt.ArrayLike, matrix2: npt.ArrayLike) -> float:

        """
        Computes the distance between two matrices/vectors.

        Args
        ----
        - **matrix1 (np.ndarray)**: The first matrix/vector.
        - **matrix2 (np.ndarray)**: The second matrix/vector.

        Returns
        -------
        - **float**: The distance between `matrix1` and `matrix2`.
        """

        pass


class CosineDistance(DistanceMetric):

    """Cosine distance between two vectors."""

    def compute(self, matrix1: npt.ArrayLike, matrix2: npt.ArrayLike) -> float:
        return float(sp.distance.cosine(matrix1, matrix2))


class EuclideanDistance(DistanceMetric):

    """Euclidean (straight-line) distance between two vectors."""

    def compute(self, matrix1: npt.ArrayLike, matrix2: npt.ArrayLike) -> float:
        return float(sp.distance.euclidean(matrix1, matrix2))


class ManhattanDistance(DistanceMetric):

    """Manhattan (L1 / city block) distance between two vectors."""

    def compute(self, matrix1: npt.ArrayLike, matrix2: npt.ArrayLike) -> float:
        return float(sp.distance.cityblock(matrix1, matrix2))


class JaccardDistance(DistanceMetric):

    """Jaccard distance between two boolean/binary vectors."""

    def compute(self, matrix1: npt.ArrayLike, matrix2: npt.ArrayLike) -> float:
        return float(sp.distance.jaccard(matrix1, matrix2))


class HammingDistance(DistanceMetric):

    """Hamming distance: the proportion of differing elements."""

    def compute(self, matrix1: npt.ArrayLike, matrix2: npt.ArrayLike) -> float:
        return float(sp.distance.hamming(matrix1, matrix2))


class ChebyshevDistance(DistanceMetric):

    """Chebyshev distance: the maximum absolute difference between elements."""

    def compute(self, matrix1: npt.ArrayLike, matrix2: npt.ArrayLike) -> float:
        return float(sp.distance.chebyshev(matrix1, matrix2))


class MinkowskiDistance(DistanceMetric):

    """
    Minkowski distance, a generalization of the Euclidean and Manhattan
    distances.

    Attributes
    ----------
    - **p (int)**: The order of the Minkowski distance (default: 3).
    """

    def __init__(self, p: int = 3) -> None:
        if p <= 0:
            raise ValueError("Order p must be a positive integer.")
        self.p = p

    def compute(self, matrix1: npt.ArrayLike, matrix2: npt.ArrayLike) -> float:
        return float(sp.distance.minkowski(matrix1, matrix2, p=self.p))


class CanberraDistance(DistanceMetric):

    """Canberra distance: a weighted Manhattan distance sensitive to small values."""

    def compute(self, matrix1: npt.ArrayLike, matrix2: npt.ArrayLike) -> float:
        return float(sp.distance.canberra(matrix1, matrix2))


class BrayCurtisDistance(DistanceMetric):

    """Bray-Curtis dissimilarity between two vectors."""

    def compute(self, matrix1: npt.ArrayLike, matrix2: npt.ArrayLike) -> float:
        return float(sp.distance.braycurtis(matrix1, matrix2))


class CorrelationDistance(DistanceMetric):

    """Correlation distance: 1 minus the Pearson correlation coefficient."""

    def compute(self, matrix1: npt.ArrayLike, matrix2: npt.ArrayLike) -> float:
        return float(sp.distance.correlation(matrix1, matrix2))


class ProcrustesDistance(DistanceMetric):

    """
    Procrustes disparity between two matrices after optimal scaling,
    translation, and rotation. Returns 0.0 (no distance) instead of running
    the test whenever either matrix has fewer than 2 rows or fewer than 2
    distinct rows, since Procrustes analysis is undefined in that case.
    """

    def compute(self, matrix1: npt.ArrayLike, matrix2: npt.ArrayLike) -> float:
        matrix1 = np.asarray(matrix1)
        matrix2 = np.asarray(matrix2)
        if (
            matrix1.shape[0] < 2 or matrix2.shape[0] < 2 or
            np.unique(matrix1, axis=0).shape[0] < 2 or
            np.unique(matrix2, axis=0).shape[0] < 2
        ):
            return 0.0

        _, _, disparity = sp.procrustes(matrix1, matrix2)
        return float(disparity)


# Name -> metric, mirroring bio/selection.py's SELECTION_METHODS registry.
# GAMIPY.fitness_method is resolved through DistanceMetric.create() (see
# GAMIPY.fitness_metric) to pick one of these; 'procrustes' is excluded from
# that selection (see DistanceMetric.create) and used internally, fixed, by
# Fitness for the covariance comparison.
DISTANCE_METRICS: dict[str, type[DistanceMetric]] = {
    'cosine': CosineDistance,
    'euclidean': EuclideanDistance,
    'manhattan': ManhattanDistance,
    'jaccard': JaccardDistance,
    'hamming': HammingDistance,
    'chebyshev': ChebyshevDistance,
    'minkowski': MinkowskiDistance,
    'canberra': CanberraDistance,
    'braycurtis': BrayCurtisDistance,
    'correlation': CorrelationDistance,
    'procrustes': ProcrustesDistance,
}
