"""Column statistics of many datasets at once, in NumPy.

``Fitness`` scores a population by comparing the statistics (mean, skewness,
covariance / correlation) of each individual's imputed dataset with those of
the observed data. Computing them with pandas, one DataFrame per individual,
took most of a GA run. ``Moments`` holds the same statistics for a stack of
same-shaped datasets (individuals x columns x rows) and computes them with a
few vectorized NumPy operations.

The rows every individual shares (the complete rows, with full-data
statistics) are summarized once; ``combine`` merges that summary with the
summary of each individual's own rows (Chan, Golub & LeVeque's pairwise
update), so a generation costs time proportional to the rows that contain a
missing value, not to the whole dataset.

Conventions follow pandas (``DataFrame.mean``, ``skew``, ``cov``, ``corr``):
bias-corrected sample skewness that is 0 for a constant column and NaN for
fewer than 3 rows, covariance with ``n - 1`` degrees of freedom, and NaN
correlations for a constant column. Results agree with pandas to floating
point rounding (about 1e-15 relative), not always to the last bit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# pandas treats sums of squared / cubed deviations below this as rounding
# noise (0) when computing skewness; used for correlations too, so a
# constant column gets NaN correlations as in pandas.
ZERO_TOLERANCE = 1e-14

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Moments:

    """
    Per-column statistics of a stack of datasets with the same number of rows.

    Attributes
    ----------
    - **count (int)**: Rows per dataset.
    - **mean (ndarray)**: Column means, shape (datasets, columns).
    - **m2 (ndarray)**: Sums of squared deviations from the mean, shape (datasets, columns).
    - **m3 (ndarray)**: Sums of cubed deviations from the mean, shape (datasets, columns).
    - **cross (ndarray)**: Sums of products of deviations, shape (datasets, columns, columns).
    """

    count: int
    mean: FloatArray
    m2: FloatArray
    m3: FloatArray
    cross: FloatArray

    @classmethod
    def of(cls, stacked: FloatArray) -> Moments:
        """The moments of each dataset in `stacked` (datasets x columns x
        rows, no missing values)."""
        count = stacked.shape[2]
        mean = stacked.mean(axis=2)
        deviations = stacked - mean[:, :, None]
        squared = deviations * deviations
        return cls(
            count=count,
            mean=mean,
            m2=squared.sum(axis=2),
            m3=(squared * deviations).sum(axis=2),
            cross=deviations @ deviations.transpose(0, 2, 1),
        )

    def combine(self, other: Moments) -> Moments:
        """The moments of each dataset of `other` with this summary's rows
        added (this summary is for one dataset, broadcast to all of `other`'s)."""
        n_a, n_b = self.count, other.count
        n = n_a + n_b
        delta = other.mean - self.mean
        mean = self.mean + delta * (n_b / n)
        m2 = self.m2 + other.m2 + delta ** 2 * (n_a * n_b / n)
        m3 = (self.m3 + other.m3 + delta ** 3 * (n_a * n_b * (n_a - n_b) / n ** 2)
              + 3 * delta * (n_a * other.m2 - n_b * self.m2) / n)
        cross = self.cross + other.cross + delta[:, :, None] * delta[:, None, :] * (n_a * n_b / n)
        return Moments(count=n, mean=mean, m2=m2, m3=m3, cross=cross)

    def skew(self) -> FloatArray:
        """Bias-corrected sample skewness per column, as ``DataFrame.skew``."""
        n = self.count
        m2 = np.where(np.abs(self.m2) < ZERO_TOLERANCE, 0.0, self.m2)
        m3 = np.where(np.abs(self.m3) < ZERO_TOLERANCE, 0.0, self.m3)
        if n < 3:
            return np.full_like(m2, np.nan)
        with np.errstate(invalid='ignore', divide='ignore'):
            result = (n * (n - 1) ** 0.5 / (n - 2)) * (m3 / m2 ** 1.5)
        return np.where(m2 == 0, 0.0, result)

    def cov(self) -> FloatArray:
        """Sample covariance matrices (``n - 1`` degrees of freedom), as ``DataFrame.cov``."""
        with np.errstate(invalid='ignore', divide='ignore'):
            return self.cross / (self.count - 1)

    def corr(self) -> FloatArray:
        """Pearson correlation matrices, NaN for a constant column, as ``DataFrame.corr``."""
        variance = np.where(self.m2 < ZERO_TOLERANCE, 0.0, self.m2)
        scale = np.sqrt(variance[:, :, None] * variance[:, None, :])
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(scale > 0, self.cross / scale, np.nan)
