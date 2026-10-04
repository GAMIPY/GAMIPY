
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from GAMIPY.bio.column_statistics import CategoricalCentralTendency, ColumnStatisticsProvider, NumericalCentralTendency
from GAMIPY.bio.distance import CosineDistance, DistanceMetric, ProcrustesDistance
from GAMIPY.bio.encoding import CategoricalEncoder
from GAMIPY.bio.moments import Moments
from GAMIPY.data.data import Data
from GAMIPY.parallel import map_chunks, thread_count
from GAMIPY.population.individual import Individual
from GAMIPY.population.population import Population


class _ImputedRows:

    """
    The encoded imputed rows (the rows with a missing value) of many
    individuals at once, in NumPy: each individual's genes (categorical labels
    as their codes) written into a copy of the encoded rows. Writing each gene
    into a DataFrame copy with ``.loc``, and computing the statistics with
    pandas, used to take most of a GA run.

    With full-data statistics, `prefix` holds the encoded complete rows; they
    are the same for every individual, so their moments are computed once
    (``prefix_moments``) and combined with each individual's.
    """

    # Upper bound on the cells of one stack of individuals (individuals x
    # columns x rows), to keep memory bounded on large datasets.
    CHUNK_CELLS = 4_000_000

    def __init__(self, encoded: pd.DataFrame, codes: dict[Any, dict[Any, int]], prefix: pd.DataFrame | None) -> None:
        self.columns = list(encoded.columns)
        self.base = np.ascontiguousarray(encoded.to_numpy(dtype=float).T)  # columns x rows
        self.codes = codes
        self._n_rows = self.base.shape[1]
        self._rows = {label: i for i, label in enumerate(encoded.index)}
        self._cols = {col: j for j, col in enumerate(self.columns)}
        # Flat index (into columns x rows) of every missing cell, by
        # Gene._m_location; a gene for any other cell is placed via _rows/_cols.
        cols, rows = np.nonzero(np.isnan(self.base))
        self._flat = {
            (encoded.index[row], self.columns[col]): col * self._n_rows + row
            for col, row in zip(cols.tolist(), rows.tolist(), strict=True)
        }
        # Gene order -> flat indices cache of _layout.
        self._last_locations: list[tuple[Any, Any]] = []
        self._last_index: list[int] = []
        self._last_coded: list[tuple[int, dict[Any, int]]] = []
        self.prefix_moments: Moments | None = None
        if prefix is not None and len(prefix):
            self.prefix_moments = Moments.of(np.ascontiguousarray(prefix.to_numpy(dtype=float).T)[None])

    @classmethod
    def build(cls, m_data: Data, encoder: CategoricalEncoder, prefix: pd.DataFrame | None) -> "_ImputedRows | None":
        """The builder for `m_data`, or None when the vectorized path doesn't
        apply (the encoder has no fixed codes for a categorical column, a
        column isn't a plain NumPy number after encoding, or the complete rows
        have a missing value) and every individual is scored with pandas."""
        if any(col not in encoder._codes for col in m_data.get_categorical_columns):
            return None
        encoded = encoder.encode(m_data).data
        if not all(isinstance(dtype, np.dtype) and dtype.kind in 'biuf' for dtype in encoded.dtypes):
            return None
        if prefix is not None:
            if list(prefix.columns) != list(encoded.columns):
                return None
            if not all(isinstance(dtype, np.dtype) and dtype.kind in 'biuf' for dtype in prefix.dtypes):
                return None
            if prefix.isna().to_numpy().any():
                return None
        return cls(encoded, encoder._codes, prefix)

    def flat_index(self, location: tuple[Any, Any]) -> int:
        """Index of the cell ``(row label, column)`` in a flattened stack entry."""
        index = self._flat.get(location)
        if index is None:
            row, col = location
            index = self._cols[col] * self._n_rows + self._rows[row]
        return index

    def encode_value(self, column: Any, value: Any) -> Any:
        """`value` as it appears in the encoded rows (a code for a categorical column)."""
        codes = self.codes.get(column)
        return value if codes is None else codes.get(value, np.nan)

    def stack(self, individuals: list[Individual]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        """
        The encoded imputed rows of `individuals` (individuals x columns x
        rows), and which of them are complete: an individual that leaves a
        missing cell empty, or whose genes can't be written (e.g. a label the
        encoder doesn't know), is left to the pandas path.
        """
        stacked = np.empty((len(individuals), *self.base.shape))
        stacked[:] = self.base
        flat = stacked.reshape(len(individuals), -1)
        complete = np.ones(len(individuals), dtype=bool)
        for i, individual in enumerate(individuals):
            try:
                # _layout first: _values uses the gene order it records.
                index = self._layout(individual)
                flat[i, index] = self._values(individual)
            except (KeyError, TypeError, ValueError):
                complete[i] = False
        complete &= ~np.isnan(flat).any(axis=1)
        return stacked, complete

    def _layout(self, individual: Individual) -> list[int]:
        """Flat index of each of `individual`'s genes. Individuals built by the
        GA share one gene order, so the index list of the last order seen is
        reused when the next individual's locations are the same."""
        locations = [gene._m_location for gene in individual.genes]
        if locations != self._last_locations:
            self._last_locations = locations
            self._last_index = [self.flat_index(location) for location in locations]
            self._last_coded = [(i, self.codes[location[1]]) for i, location in enumerate(locations)
                                if location[1] in self.codes]
        return self._last_index

    def _values(self, individual: Individual) -> list[Any]:
        """`individual`'s gene values as they appear in the encoded rows
        (categorical labels as codes), for the gene order of the last
        ``_layout`` call."""
        values = [gene.data for gene in individual.genes]
        for i, codes in self._last_coded:
            values[i] = codes.get(values[i], np.nan)
        return values

    def moments(self, stacked: npt.NDArray[np.float64]) -> Moments:
        """The moments of each imputed dataset (the prefix rows included)."""
        moments = Moments.of(stacked)
        return moments if self.prefix_moments is None else self.prefix_moments.combine(moments)


class Fitness:

    """
    Orchestrates fitness evaluation for a population of individuals: builds
    each individual's imputed dataset, and scores it against the complete
    data by comparing mean, covariance, and skewness using distance
    metrics.

    Categorical encode/decode (GAMIPY.bio.encoding.CategoricalEncoder) and
    the distance metrics themselves (GAMIPY.bio.distance) are separate
    Strategy classes that Fitness depends on rather than implements.

    The `metric` used for the mean and skewness comparisons is injectable
    (any of the registered distance metrics). The covariance
    comparison always uses `ProcrustesDistance` specifically, regardless of
    `metric`: it's the one registered metric built to compare whole
    matrices (after optimal scaling/rotation) rather than plain vectors, so
    it isn't swappable the way the mean/skewness metric is.

    `mean_score` likewise depends on the `ColumnStatisticsProvider`
    abstraction (GAMIPY.bio.column_statistics) rather than instantiating
    `NumericalColumn`/`CategoricalColumn` directly.

    Attributes
    ----------
    - **population (list)**: A list of Individual objects representing the population.
    - **m_data (Data)**: The data with missing values that needs to be imputed.
    - **c_data (Data)**: The complete data used for comparison with the imputed data.
    - **encoder (CategoricalEncoder)**: Encodes categorical columns as integer codes before scoring.
      Defaults to `CategoricalEncoder()`.
    - **metric (DistanceMetric)**: The distance metric used to compare mean and skewness vectors.
      Defaults to `CosineDistance()`. Build one from a name (e.g. GAMIPY's `fitness_method`) with
      `DistanceMetric.create(name)`.
    - **numeric_stats (ColumnStatisticsProvider)**: Provides the representative value of a numeric
      column for `mean_score`. Defaults to `NumericalCentralTendency()`.
    - **categorical_stats (ColumnStatisticsProvider)**: Provides the representative value of a
      categorical column for `mean_score`. Defaults to `CategoricalCentralTendency()`.

    Methods
    -------
    - **impute_data(individual)**: Imputes the missing data for a given individual by replacing the
      missing values with the individual's gene values.
    - **evaluate()**: Evaluates the fitness of each individual in the population by calculating
      their scores based on the imputed data.
    - **_calculate_score(m_data, c_data)**: Calculates the fitness score for the imputed data by
      comparing it with the complete data using mean, covariance, skewness, and distance metrics.
    - **covariance_score(data)**: Calculates the covariance of the data.
    - **skewness_score(data)**: Calculates the skewness of the data.
    - **mean_score(data)**: Calculates the mean of the data.

    """

    def __init__(
        self,
        population: Population | None = None,
        m_data: Data | None = None,
        c_data: Data | None = None,
        encoder: CategoricalEncoder | None = None,
        metric: DistanceMetric | None = None,
        numeric_stats: ColumnStatisticsProvider | None = None,
        categorical_stats: ColumnStatisticsProvider | None = None,
        reference: Data | None = None,
        full_data_statistics: bool = False,
        standardize: bool = False,
        n_jobs: int | None = 1,
    ) -> None:
        """
        Initializes the Fitness class with a population of individuals and the data for imputation.

        Args
        ----
        - **population (list[Individual])**: A list of Individual objects representing the
          population. Default is an empty list.
        - **m_data (Data)**: The data with missing values that needs to be imputed. Default is None.
        - **c_data (Data)**: The complete data used for comparison with the imputed data. Default is None.
        - **encoder (CategoricalEncoder)**: Encodes categorical columns as integer codes before
          scoring. Default is a new `CategoricalEncoder()`.
        - **metric (DistanceMetric)**: The distance metric used to compare mean and skewness vectors
          (see class docstring). Default is a new `CosineDistance()`.
        - **numeric_stats (ColumnStatisticsProvider)**: Provides the representative value of a
          numeric column for `mean_score`. Default is a new `NumericalCentralTendency()`.
        - **categorical_stats (ColumnStatisticsProvider)**: Provides the representative value of a
          categorical column for `mean_score`. Default is a new `CategoricalCentralTendency()`.
        - **reference (Data | None)**: Already-encoded data the statistics are compared against.
          Default: the encoded ``c_data`` (the complete rows).
        - **full_data_statistics (bool)**: Score the statistics of the whole imputed dataset (the
          complete rows plus the imputed rows) instead of the imputed rows alone. Default False.
        - **standardize (bool)**: Use scale-free terms: z-scored mean differences, skewness
          differences and correlation-matrix differences, each averaged over columns. Default False.
        - **n_jobs (int | None)**: Threads computing the statistics of the individuals, see
          ``GAMIPY.parallel.thread_count``. The scores don't depend on it. Default 1.

        """

        self.population = population
        # Optional so a Fitness can be built just to score two datasets
        # directly (_calculate_score); evaluate()/impute_data() need them.
        self.m_data = m_data
        self.c_data = c_data
        self.encoder = encoder if encoder is not None else CategoricalEncoder()
        self.metric = metric if metric is not None else CosineDistance()
        self.numeric_stats = numeric_stats if numeric_stats is not None else NumericalCentralTendency()
        self.categorical_stats = categorical_stats if categorical_stats is not None else CategoricalCentralTendency()
        self.reference = reference
        self.full_data_statistics = full_data_statistics
        self.standardize = standardize
        self.n_jobs = n_jobs

    def impute_data(self, individual: Individual) -> pd.DataFrame:
        """
        Imputes the missing data for a given individual by replacing the missing values with the
        individual's gene values.

        Args
        ----
        - **individual (Individual)**: The individual whose genes will be used to impute the missing data in the m_data.

        Returns
        -------
        - **imputed_data (Data)**: A DataFrame containing the imputed data where the missing values
          in m_data are replaced with the corresponding gene values from the individual.

        """

        if self.m_data is None:
            raise ValueError("Fitness.impute_data() needs m_data.")
        imputed_data = self.m_data.data.copy()
        for gene in individual.genes:
            imputed_data.loc[gene._m_location[0], gene._m_location[1]] = gene.data
        return imputed_data

    def evaluate(self, rescore: bool = True) -> None:
        """
        Scores every individual in the population and stores the result with
        ``individual.set_score``. With ``rescore=False``, individuals that
        already have a score are skipped: the GA runner uses this, since a
        survivor's genes and the scoring context don't change between
        generations.

        For each individual, its genes are written into a copy of ``m_data``
        (the rows with missing values), categorical columns are encoded as
        integer codes (``c_data`` is encoded once per call), and the result is
        compared with ``c_data`` (the complete rows) by ``_calculate_score``::

            score = metric(means) + metric(skewness) + procrustes(covariances)

        or, with ``standardize``, by ``_standardized_score``. Every term is a
        distance, so a lower score is better; a statistic that is undefined
        for the data (e.g. skewness of fewer than 3 rows) is left out.
        Nothing is decoded or written back: the population's scores are the
        only side effect.

        The statistics of all individuals are computed together in NumPy
        (see ``GAMIPY.bio.moments``); they agree with the pandas computation
        of ``_calculate_score`` / ``_standardized_score`` to floating point
        rounding. Individuals the vectorized path can't handle (e.g. one that
        leaves a missing cell empty) are scored with pandas.

        Raises
        ------
        - **ValueError**: If population, m_data or c_data is missing.
        """

        # self.c_data doesn't change across individuals within one evaluate() call,
        # so encode it once here rather than re-encoding (and pointlessly
        # decoding back) on every iteration of the loop below.
        if self.population is None or self.m_data is None or self.c_data is None:
            raise ValueError("Fitness.evaluate() needs population, m_data and c_data.")
        encoded_c_data = self.encoder.encode(self.c_data)
        reference = self.reference if self.reference is not None else encoded_c_data
        reference_summary = self._summary(reference.data) if self.standardize else None
        pending = [ind for ind in self.population.population if rescore or not ind.is_evaluated]
        rows = _ImputedRows.build(self.m_data, self.encoder,
                                  encoded_c_data.data if self.full_data_statistics else None)
        scored: dict[int, float] = {}
        if rows is not None and pending:
            scored = self._vectorized_scores(rows, reference, reference_summary, pending)
        for i, individual in enumerate(pending):
            if i in scored:
                individual.set_score(scored[i])
                continue
            m_data = self.encoder.encode(Data(self.impute_data(individual)))
            if self.full_data_statistics:
                m_data = Data(pd.concat([encoded_c_data.data, m_data.data], ignore_index=True))
            if reference_summary is not None:
                score = self._standardized_score(m_data.data, reference_summary)
            else:
                score = self._calculate_score(m_data, reference)
            individual.set_score(score)

    def _vectorized_scores(
        self, rows: _ImputedRows, reference: Data, reference_summary: dict[str, Any] | None,
        individuals: list[Individual],
    ) -> dict[int, float]:
        """Scores of `individuals` computed together in NumPy, by position in
        the list; individuals the vectorized path can't handle are left out."""
        columns = rows.columns
        if reference_summary is not None:
            target: dict[str, Any] = {
                'mean': reference_summary['mean'].reindex(columns).to_numpy(dtype=float),
                'std': reference_summary['std'].reindex(columns).to_numpy(dtype=float),
                'skew': reference_summary['skew'].reindex(columns).to_numpy(dtype=float),
                'corr': reference_summary['corr'].reindex(index=columns, columns=columns).to_numpy(dtype=float),
            }
        else:
            # The reference statistics don't depend on the individual.
            target = {
                'mean': np.asarray(self.mean_score(reference), dtype=float),
                'cov': self.covariance_score(reference),
                'skew': self.skewness_score(reference),
            }
        scores: dict[int, float] = {}
        per_chunk = max(1, rows.CHUNK_CELLS // max(1, rows.base.size))
        threads = thread_count(self.n_jobs)
        for start in range(0, len(individuals), per_chunk):
            # stack() updates rows' gene-order cache, so it runs here. The
            # moments run in threads; the scores are computed from them joined
            # back together, as without threads (BLAS rounding can depend on
            # where a vector sits in memory, so scoring each thread's arrays
            # could differ in the last bit).
            stacked, complete = rows.stack(individuals[start:start + per_chunk])
            if not complete.any():
                continue
            stacked = stacked[complete]

            def moments_of(part: slice, stacked: npt.NDArray[np.float64] = stacked) -> Moments:
                return rows.moments(stacked[part])

            moments = Moments.concatenate(map_chunks(moments_of, len(stacked), threads))
            if reference_summary is not None:
                values = self._standardized_scores(moments, target)
            else:
                values = self._moment_scores(moments, target)
            for position, value in zip(np.flatnonzero(complete).tolist(), values.tolist(), strict=True):
                scores[start + position] = value
        return scores

    @staticmethod
    def _standardized_scores(moments: Moments, target: dict[str, Any]) -> npt.NDArray[np.float64]:
        """``_standardized_score`` for every dataset in `moments` at once."""
        with np.errstate(invalid='ignore', divide='ignore'):
            mean_gap = np.abs(moments.mean - target['mean']) / target['std']
        skew_gap = np.abs(moments.skew() - target['skew'])
        corr_gap = np.abs(moments.corr() - target['corr'])
        corr_gap = corr_gap[:, ~np.eye(corr_gap.shape[1], dtype=bool)]
        score = np.zeros(len(mean_gap))
        for gap in (mean_gap, skew_gap, corr_gap):
            finite = np.isfinite(gap)
            count = finite.sum(axis=1)
            total = np.where(finite, gap, 0.0).sum(axis=1)
            score += np.where(count > 0, total / np.maximum(count, 1), 0.0)
        return score

    def _moment_scores(self, moments: Moments, target: dict[str, Any]) -> npt.NDArray[np.float64]:
        """``_calculate_score`` for every dataset in `moments`."""
        scores = []
        for mean, covariance, skew in zip(moments.mean, moments.cov(), moments.skew(), strict=True):
            score = self.metric.compute(mean, target['mean'])
            if self._is_defined(skew, target['skew']):
                score += self.metric.compute(skew, target['skew'])
            if self._is_defined(covariance, target['cov']):
                score += ProcrustesDistance().compute(covariance, target['cov'])
            scores.append(score)
        return np.asarray(scores, dtype=float)

    def _calculate_score(self, m_data: Data, c_data: Data) -> float:
        """
        Calculates the fitness score for the imputed data by comparing it with the complete data
        using mean covariance, skewness, and distance metrics.

        Args
        ----
        - **m_data (Data)**: The imputed data with missing values replaced by the individual's gene values.
        - **c_data (Data)**: The complete data used for comparison with the imputed data.

        Returns
        -------
        - **score (float)**: The fitness score calculated based on the comparison of the imputed
          data with the complete data.
        """

        mean_m, mean_c = self.mean_score(m_data), self.mean_score(c_data)
        covariance_m, covariance_c = self.covariance_score(m_data), self.covariance_score(c_data)
        skewness_m, skewness_c = self.skewness_score(m_data), self.skewness_score(c_data)
        # Every term is a distance (0 = imputed data statistically identical
        # to the complete data), so a lower score is better - the direction
        # Selection and best_individual optimise. (Before 2026-09-28 the mean
        # and skewness terms were `1 - distance`, i.e. similarities, while
        # the covariance term was a distance, so no single direction was
        # right for all three.)
        score = self.metric.compute(mean_m, mean_c)
        # Skewness needs >= 3 rows and covariance >= 2 (e.g. a dataset with a
        # single missing row gives a one-row m_data), and a constant column
        # has no defined skewness either. An undefined statistic used to turn
        # every score into NaN (skewness) or crash ProcrustesDistance
        # (covariance); it is now left out of the score instead. When all
        # three are defined, the score is unchanged.
        if self._is_defined(skewness_m, skewness_c):
            score += self.metric.compute(skewness_m, skewness_c)
        if self._is_defined(covariance_m, covariance_c):
            score += ProcrustesDistance().compute(covariance_m, covariance_c)

        return score

    @staticmethod
    def _summary(frame: pd.DataFrame) -> dict[str, Any]:
        """Means, standard deviations, skewness and correlations of `frame`
        (missing values skipped), for the standardized score."""
        return {
            'mean': frame.mean(),
            'std': frame.std().replace(0, np.nan),
            'skew': frame.skew(),
            'corr': frame.corr(),
        }

    def _standardized_score(self, frame: pd.DataFrame, reference: dict[str, Any]) -> float:
        """
        Scale-free distance between `frame` and a reference summary: the mean
        absolute z-scored mean difference, the mean absolute skewness
        difference and the mean absolute correlation difference (off the
        diagonal). Terms (or columns) that are undefined for the data are left
        out, like in ``_calculate_score``.
        """
        mean_gap = ((frame.mean() - reference['mean']).abs() / reference['std']).to_numpy(dtype=float)
        skew_gap = (frame.skew() - reference['skew']).abs().to_numpy(dtype=float)
        corr_gap = (frame.corr() - reference['corr']).abs().to_numpy(dtype=float)
        if corr_gap.size:
            corr_gap = corr_gap[~np.eye(corr_gap.shape[0], dtype=bool)]
        score = 0.0
        for gap in (mean_gap, skew_gap, corr_gap):
            finite = gap[np.isfinite(gap)]
            if finite.size:
                score += float(finite.mean())
        return score

    @staticmethod
    def _is_defined(*stats: npt.ArrayLike) -> bool:
        return all(np.isfinite(np.asarray(stat, dtype=float)).all() for stat in stats)

    def covariance_score(self, data: Data) -> npt.NDArray[Any]:
        """
        Calculates the covariance of the data.

        Args
        ----
        - **data (Data)**: The data for which the covariance is to be calculated.

        Returns
        -------
        - **np.ndarray**: A NumPy array containing the covariance matrix of the data.
        """

        data_frame = data.to_dataframe
        return np.array(data_frame.cov())

    def skewness_score(self, data: Data) -> npt.NDArray[Any]:
        """
        Calculates the skewness of the data.

        Args
        ----
        - **data (Data)**: The data for which the skewness is to be calculated.

        Returns
        -------
        - **np.ndarray**: A NumPy array containing the skewness values of the data
        """

        data_frame = data.to_dataframe
        return np.array(data_frame.skew())

    def mean_score(self, data: Data) -> list[Any]:
        """
        Calculates the mean or mode of each column in the data. For numeric columns, it calculates
        the mean value, and for categorical columns, it calculates the mode.

        Args
        ----
        - **data (Data)**: The data for which the mean or mode is to be calculated.

        Returns
        -------
        - **list**: A list containing the mean values for numeric columns and mode values for categorical columns.
        """

        mean_scores = []

        for col in data.get_columns:
            if col in data.get_numeric_columns:
                mean_scores.append(self.numeric_stats.central_value(data, col))
            elif col in data.get_categorical_columns:
                mean_scores.append(self.categorical_stats.central_value(data, col))
            else:
                raise ValueError(f"Column '{col}' is neither numeric nor categorical.")
        return mean_scores


# Observed cells of each target column: rows and true values (standardized
# values, or class indices for a categorical column).
TargetCells = dict[Any, tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]]


class ModelFitness:

    """
    Evaluates candidates of the model-based GA (``representation='model'``,
    run by ``GAMIPY.model_search.ModelSearch``). A candidate holds one linear
    model per target column (a column with a missing value); the models
    predict the target from the other columns of the same row. The fitness of
    a candidate is its error on the observed cells of every target: the
    squared standardized error for a numerical target, and for a categorical
    one the cross-entropy of the softmax of its class scores (``loss='log'``)
    or the misclassification rate (``loss='zero_one'``). Lower is better.

    Candidates are passed as arrays: ``weights`` (candidates x outputs x
    features) and ``bias`` (candidates x outputs). A numerical target has one
    output and a categorical target one output per class; a numerical input
    column has one feature slot and a categorical one a slot per class
    (one-hot).

    Attributes
    ----------
    - **targets (list)**: The target columns, in output order.
    - **numeric (set)**: The numerical columns.
    - **feature_slots (dict)**: Feature positions of each column.
    - **output_slots (dict)**: Output positions of each target.
    - **chain_rounds (int)**: Rounds of re-predicting missing inputs from the models' own outputs (default 3).
    - **n_jobs (int | None)**: Threads that score the candidates (see ``GAMIPY.parallel``).
    - **training_loss (str)**: Loss of categorical targets in the fitness: ``'log'`` (a smooth signal the
      GA can climb) or ``'zero_one'``.
    - **selection_loss (str)**: Loss of categorical targets on the validation cells, used to pick the result.
    """

    training_loss = 'log'
    selection_loss = 'log'

    def __init__(
        self,
        targets: list[Any],
        numeric: list[Any],
        feature_slots: dict[Any, list[int]],
        output_slots: dict[Any, list[int]],
        chain_rounds: int = 3,
        n_jobs: int | None = None,
    ) -> None:
        self.targets = targets
        self.numeric = set(numeric)
        self.feature_slots = feature_slots
        self.output_slots = output_slots
        self.chain_rounds = chain_rounds
        self.n_jobs = n_jobs

    def predict(
        self,
        weights: npt.NDArray[np.float64],
        bias: npt.NDArray[np.float64],
        features: npt.NDArray[np.float64],
        missing: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.float64]:
        """
        Outputs of every candidate's models (candidates x rows x outputs),
        after ``chain_rounds`` rounds of filling the missing inputs with the
        models' own predictions (a categorical input takes the one-hot code
        of its predicted class).

        Args
        ----
        - **weights (np.ndarray)**: Candidates x outputs x features.
        - **bias (np.ndarray)**: Candidates x outputs.
        - **features (np.ndarray)**: Rows x features, missing inputs set to 0.
        - **missing (np.ndarray)**: Rows x features, True where the input is missing.
        """
        state = np.repeat(features[None], len(weights), axis=0)
        for _ in range(self.chain_rounds):
            outputs = state @ weights.transpose(0, 2, 1) + bias[:, None, :]
            for col in self.targets:
                rows = missing[:, self.feature_slots[col][0]]
                if not rows.any():
                    continue
                out = self.output_slots[col]
                if col in self.numeric:
                    state[:, rows, self.feature_slots[col][0]] = outputs[:, rows, out[0]]
                else:
                    scores = outputs[:, rows][:, :, out]
                    one_hot = np.eye(len(out))[scores.argmax(axis=2)]
                    block = state[:, rows]
                    block[:, :, self.feature_slots[col]] = one_hot
                    state[:, rows] = block
        result: npt.NDArray[np.float64] = state @ weights.transpose(0, 2, 1) + bias[:, None, :]
        return result

    def column_errors(self, outputs: npt.NDArray[np.float64], cells: TargetCells, loss: str) -> npt.NDArray[np.float64]:
        """
        Error of every candidate on each target (candidates x targets; NaN for
        a target without cells in `cells`).

        Args
        ----
        - **outputs (np.ndarray)**: Candidates x rows x outputs, from ``predict``.
        - **cells (dict)**: The cells to score, see ``TargetCells``.
        - **loss (str)**: ``'log'`` or ``'zero_one'``, for the categorical targets.
        """
        errors = np.full((len(outputs), len(self.targets)), np.nan)
        for t, col in enumerate(self.targets):
            rows, truth = cells.get(col, (np.array([], dtype=np.intp), np.array([])))
            if not len(rows):
                continue
            out = self.output_slots[col]
            if col in self.numeric:
                errors[:, t] = ((outputs[:, rows, out[0]] - truth) ** 2).mean(axis=1)
            elif loss == 'log':
                scores = outputs[:, rows][:, :, out]
                scores = scores - scores.max(axis=2, keepdims=True)
                log_probability = scores - np.log(np.exp(scores).sum(axis=2, keepdims=True))
                picked = np.take_along_axis(log_probability, truth.astype(int)[None, :, None], axis=2)[:, :, 0]
                errors[:, t] = -picked.mean(axis=1)
            else:
                errors[:, t] = (outputs[:, rows][:, :, out].argmax(axis=2) != truth).mean(axis=1)
        return errors

    def evaluate(
        self,
        weights: npt.NDArray[np.float64],
        bias: npt.NDArray[np.float64],
        features: npt.NDArray[np.float64],
        missing: npt.NDArray[np.bool_],
        training: TargetCells,
        validation: TargetCells,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Training and validation errors of every candidate (candidates x
        targets each; the training errors twice without validation cells),
        computed in ``n_jobs`` threads over chunks of candidates. The training
        errors are the fitness; the validation errors pick the result.
        """

        def score(part: slice) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            outputs = self.predict(weights[part], bias[part], features, missing)
            errors = self.column_errors(outputs, training, self.training_loss)
            judged = self.column_errors(outputs, validation, self.selection_loss) if validation else errors
            return errors, judged

        parts = map_chunks(score, len(weights), thread_count(self.n_jobs))
        return (np.concatenate([errors for errors, _ in parts]),
                np.concatenate([judged for _, judged in parts]))
