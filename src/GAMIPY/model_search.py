"""A GA that evolves prediction models instead of cell values.

The cell-value GA searches one value per missing cell: hundreds to thousands
of free parameters against a few dozen validation cells, which it overfits.
``ModelSearch`` searches a much smaller space: for every column with a
missing value, a linear model that predicts it from the other columns of the
same row (standardized numerical columns and one-hot categorical columns;
one score per class for a categorical target). The number of parameters
depends on the number of columns, not on the number of missing cells.

Everything comes from the data and the GA itself: models start random, are
scored by how well they predict observed cells, and missing inputs are filled
with the models' own predictions (starting from the neutral value 0 of the
standardized scale and refined over a few rounds). No other imputation
method is involved.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from GAMIPY.config import GAConfig
from GAMIPY.data.data import Data
from GAMIPY.parallel import map_chunks, thread_count

FloatArray = npt.NDArray[np.float64]

# Rounds of re-predicting missing inputs from the models' own outputs.
CHAIN_ROUNDS = 3
# Share of each new generation that is random models, and of elites kept.
IMMIGRANT_SHARE = 0.1
ELITE_SHARE = 0.1
# Standard deviation of the random initial weights. Small: a model starts
# close to its constant and only grows the weights of inputs that help;
# large random weights drowned the constant in noise (categorical accuracy
# fell below the majority class).
INIT_SCALE = 0.05


class ModelSearch:

    """
    Imputes a dataset with per-column linear models whose parameters a GA
    evolves. Used by ``GAMIPY`` with ``representation='model'``.

    Attributes
    ----------
    - **config (GAConfig)**: The hyperparameters (``population_size``, ``generations``,
      ``tournament_size``, ``crossover_rate``, ``mutation_rate``, ``mutation_scale``,
      ``validation_fraction``, ``patience``, ``n_jobs``).
    - **data (Data)**: The dataset as given.
    - **targets (list)**: The columns with a missing value, whose models are evolved.
    - **generations_run (int)**: Generations evaluated by the last ``run()``.
    - **best_validation_error (float | None)**: Validation error of the returned models.
    - **history (list)**: One dict per generation of the last ``run()``: ``generation``,
      ``best_training_error`` (lowest total training error in that generation) and
      ``best_validation_error`` (mean over targets of the best validation error so far; NaN
      without validation cells).
    """

    # How categorical targets are scored: 'log' (cross-entropy of the class
    # scores' softmax: a smooth signal the GA can climb) or 'zero_one'
    # (misclassification rate), for the fitness and for picking the result.
    training_loss = 'log'
    selection_loss = 'log'

    def __init__(self, config: GAConfig, data: Data) -> None:
        self.config = config
        self.data = data
        frame = data.data
        self.numeric = list(data.get_numeric_columns)
        self.categorical = [c for c in frame.columns if c not in self.numeric]
        self.classes = {c: list(frame[c].dropna().unique()) for c in self.categorical}
        self.center = {c: float(frame[c].mean()) for c in self.numeric}
        # A constant column, or one with a single observed value (std NaN),
        # is scaled by 1.
        self.scale = {c: self._usable_scale(frame[c].std()) for c in self.numeric}
        self.targets = [c for c in frame.columns if frame[c].isna().any()]
        self.generations_run = 0
        self.best_validation_error: float | None = None
        self.history: list[dict[str, float]] = []
        self._layout()

    @staticmethod
    def _usable_scale(std: Any) -> float:
        value = float(std)
        return value if np.isfinite(value) and value > 0 else 1.0

    # --- encoding ---------------------------------------------------------

    def _layout(self) -> None:
        """Feature and output positions: each column occupies one feature
        slot (numerical) or one per class (categorical); each target one
        output (numerical) or one per class."""
        self.feature_slots: dict[Any, list[int]] = {}
        position = 0
        for col in self.data.data.columns:
            width = 1 if col in self.numeric else len(self.classes[col])
            self.feature_slots[col] = list(range(position, position + width))
            position += width
        self.n_features = position
        self.output_slots: dict[Any, list[int]] = {}
        position = 0
        for col in self.targets:
            width = 1 if col in self.numeric else len(self.classes[col])
            self.output_slots[col] = list(range(position, position + width))
            position += width
        self.n_outputs = position
        # A model never sees its own column.
        self.allowed = np.ones((self.n_outputs, self.n_features))
        for col in self.targets:
            for out in self.output_slots[col]:
                self.allowed[out, self.feature_slots[col]] = 0.0

    def _encode(self, frame: pd.DataFrame) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
        """Features (rows x features; missing = 0, the neutral value) and which
        feature slots are missing."""
        n = len(frame)
        features = np.zeros((n, self.n_features))
        missing = np.zeros((n, self.n_features), dtype=bool)
        for col, slots in self.feature_slots.items():
            values = frame[col]
            absent = values.isna().to_numpy()
            if col in self.numeric:
                standardized = (values.astype(float) - self.center[col]) / self.scale[col]
                features[:, slots[0]] = np.where(absent, 0.0, standardized)
            else:
                for slot, label in zip(slots, self.classes[col], strict=True):
                    features[:, slot] = (values == label).to_numpy(dtype=float)
            missing[:, slots] = absent[:, None]
        return features, missing

    # --- prediction -------------------------------------------------------

    def _predict(self, weights: FloatArray, bias: FloatArray, features: FloatArray,
                 missing: npt.NDArray[np.bool_]) -> FloatArray:
        """Outputs of every model (candidates x rows x outputs), after
        ``CHAIN_ROUNDS`` rounds of filling missing inputs with the models' own
        predictions."""
        state = np.repeat(features[None], len(weights), axis=0)
        for _ in range(CHAIN_ROUNDS):
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
        return state @ weights.transpose(0, 2, 1) + bias[:, None, :]

    def _column_errors(
        self, outputs: FloatArray, cells: dict[Any, tuple[npt.NDArray[np.intp], FloatArray]], loss: str,
    ) -> FloatArray:
        """Error of every candidate on each target (candidates x targets, NaN
        for a target without cells in `cells`): squared standardized error
        for numerical targets; for categorical ones, the cross-entropy of the
        softmax of the class scores (``loss='log'``) or the misclassification
        rate (``loss='zero_one'``)."""
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

    def _evaluate(
        self, weights: FloatArray, bias: FloatArray, features: FloatArray, missing: npt.NDArray[np.bool_],
        training: dict[Any, tuple[npt.NDArray[np.intp], FloatArray]],
        validation: dict[Any, tuple[npt.NDArray[np.intp], FloatArray]],
    ) -> tuple[FloatArray, FloatArray]:
        """Training and validation errors of every candidate (candidates x
        targets each; the training errors twice without validation cells),
        computed in ``config.n_jobs`` threads over chunks of candidates."""

        def score(part: slice) -> tuple[FloatArray, FloatArray]:
            outputs = self._predict(weights[part], bias[part], features, missing)
            errors = self._column_errors(outputs, training, self.training_loss)
            judged = self._column_errors(outputs, validation, self.selection_loss) if validation else errors
            return errors, judged

        parts = map_chunks(score, len(weights), thread_count(self.config.n_jobs))
        return (np.concatenate([errors for errors, _ in parts]),
                np.concatenate([judged for _, judged in parts]))

    def _cells(self, frame: pd.DataFrame) -> dict[Any, tuple[npt.NDArray[np.intp], FloatArray]]:
        """Observed cells of each target in `frame`: rows and true values
        (standardized, or class index)."""
        cells = {}
        for col in self.targets:
            rows = np.flatnonzero(frame[col].notna().to_numpy())
            values = frame[col].iloc[rows]
            if col in self.numeric:
                truth = ((values.astype(float) - self.center[col]) / self.scale[col]).to_numpy()
            else:
                truth = np.array([self.classes[col].index(v) for v in values], dtype=float)
            cells[col] = (rows, truth)
        return cells

    # --- GA ---------------------------------------------------------------

    def _random(self, count: int) -> tuple[FloatArray, FloatArray]:
        weights = np.random.normal(0.0, INIT_SCALE, (count, self.n_outputs, self.n_features)) * self.allowed
        bias = np.random.normal(0.0, INIT_SCALE, (count, self.n_outputs))
        return weights, bias

    def _tournament(self, scores: FloatArray) -> int:
        entrants = np.random.choice(len(scores), size=min(self.config.tournament_size, len(scores)), replace=False)
        return int(entrants[np.argmin(scores[entrants])])

    def _next_generation(
        self, weights: FloatArray, bias: FloatArray, errors: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """
        The next population. The fitness is a sum over target columns and each
        target has its own model, so selection works per target (cooperative
        coevolution): a child's model for a target comes from tournaments on
        that target's error, so good models for different columns found in
        different individuals are combined. Kept as they are: the individuals
        with the lowest total error, and one individual assembled from the
        best model of every target. Then random immigrants.
        """
        size = len(weights)
        filled = np.nan_to_num(errors, nan=0.0)
        order = np.argsort(filled.sum(axis=1), kind='stable')
        elite = order[:max(1, int(ELITE_SHARE * size))]
        assembled_w, assembled_b = weights[order[0]].copy(), bias[order[0]].copy()
        for t, col in enumerate(self.targets):
            best = int(np.argmin(filled[:, t]))
            slots = self.output_slots[col]
            assembled_w[slots], assembled_b[slots] = weights[best, slots], bias[best, slots]
        immigrants_w, immigrants_b = self._random(int(IMMIGRANT_SHARE * size))
        count = max(0, size - len(elite) - 1 - len(immigrants_w))
        children_w = np.empty((count, self.n_outputs, self.n_features))
        children_b = np.empty((count, self.n_outputs))
        for i in range(count):
            w: FloatArray = np.empty((self.n_outputs, self.n_features))
            c: FloatArray = np.empty(self.n_outputs)
            for t, col in enumerate(self.targets):
                slots = self.output_slots[col]
                a, b = self._tournament(filled[:, t]), self._tournament(filled[:, t])
                if np.random.random() < self.config.crossover_rate:
                    share = np.random.random()
                    w[slots] = share * weights[a, slots] + (1 - share) * weights[b, slots]
                    c[slots] = share * bias[a, slots] + (1 - share) * bias[b, slots]
                else:
                    w[slots], c[slots] = weights[a, slots], bias[a, slots]
            mutated = np.random.random(w.shape) < self.config.mutation_rate
            w = np.where(mutated, w + np.random.normal(0.0, self.config.mutation_scale, w.shape), w) * self.allowed
            mutated_b = np.random.random(c.shape) < self.config.mutation_rate
            c = np.where(mutated_b, c + np.random.normal(0.0, self.config.mutation_scale, c.shape), c)
            children_w[i], children_b[i] = w, c
        return (np.concatenate([weights[elite], assembled_w[None], immigrants_w, children_w]),
                np.concatenate([bias[elite], assembled_b[None], immigrants_b, children_b]))

    def run(self) -> pd.DataFrame:
        """Evolves the models and returns the imputed DataFrame.

        Raises
        ------
        - **ValueError**: If a column has no observed value at all (nothing to learn it from).
        """
        frame = self.data.data
        empty = [c for c in frame.columns if frame[c].notna().sum() == 0]
        if empty:
            raise ValueError(f"Column '{empty[0]}' has no observed values to impute from.")
        result = frame.copy()
        self.generations_run = 0
        self.history = []
        if not self.targets:
            return result
        # Validation cells: hidden from the models' inputs and training cells.
        working = frame.copy()
        validation: dict[Any, tuple[npt.NDArray[np.intp], FloatArray]] = {}
        if self.config.validation_fraction > 0:
            for col in self.targets:
                observed = np.flatnonzero(frame[col].notna().to_numpy())
                count = min(round(self.config.validation_fraction * len(observed)), len(observed) - 2)
                if count < 1:
                    continue
                rows = np.sort(np.random.choice(observed, size=count, replace=False))
                working.iloc[rows, list(frame.columns).index(col)] = np.nan
            truth_cells = self._cells(frame)
            for col, (rows, truth) in truth_cells.items():
                hidden = working[col].isna().to_numpy()[rows]
                validation[col] = (rows[hidden], truth[hidden])
        features, missing = self._encode(working)
        training = self._cells(working)

        weights, bias = self._random(max(4, self.config.population_size))
        # Per target: (validation error, its model's weights, its bias), the
        # best seen in any generation; the result assembles these models.
        best_models: list[tuple[float, FloatArray, FloatArray] | None] = [None] * len(self.targets)
        since_improvement = 0
        for _ in range(self.config.generations):
            errors, judged = self._evaluate(weights, bias, features, missing, training, validation)
            self.generations_run += 1
            improved = False
            for t, col in enumerate(self.targets):
                if np.isnan(judged[:, t]).all():
                    judged_t = np.nan_to_num(errors[:, t], nan=0.0)
                else:
                    judged_t = judged[:, t]
                leader = int(np.nanargmin(judged_t))
                current = best_models[t]
                if current is None or judged_t[leader] < current[0] - 1e-12:
                    slots = self.output_slots[col]
                    best_models[t] = (float(judged_t[leader]), weights[leader, slots].copy(),
                                      bias[leader, slots].copy())
                    improved = True
            since_improvement = 0 if improved else since_improvement + 1
            self.history.append({
                'generation': self.generations_run,
                'best_training_error': float(np.min(np.nansum(errors, axis=1))),
                'best_validation_error': (float(np.mean([m[0] for m in best_models if m is not None]))
                                          if validation else float('nan')),
            })
            if self.config.patience and since_improvement >= self.config.patience:
                break
            weights, bias = self._next_generation(weights, bias, errors)

        final_w = np.zeros((self.n_outputs, self.n_features))
        final_b = np.zeros(self.n_outputs)
        for t, col in enumerate(self.targets):
            model = best_models[t]
            assert model is not None
            final_w[self.output_slots[col]], final_b[self.output_slots[col]] = model[1], model[2]
        if validation:
            self.best_validation_error = float(np.mean([m[0] for m in best_models if m is not None]))
        best = (0.0, final_w, final_b)
        features, missing = self._encode(frame)
        outputs = self._predict(best[1][None], best[2][None], features, missing)[0]
        for col in self.targets:
            rows = frame[col].isna().to_numpy()
            out = self.output_slots[col]
            if col in self.numeric:
                seen = frame[col].dropna().astype(float)
                values = outputs[rows, out[0]] * self.scale[col] + self.center[col]
                result.loc[rows, col] = np.clip(values, seen.min(), seen.max())
            else:
                labels = np.array(self.classes[col], dtype=object)
                result.loc[rows, col] = labels[outputs[rows][:, out].argmax(axis=1)]
        return result
