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

The genetic operators live in the Bio module, like those of the cell-value
GA: ``ModelFitness`` (prediction and errors), ``ColumnTournament``
(selection), ``ArithmeticModelCrossover`` and ``GaussianModelMutation``.
``ModelSearch`` encodes the data, holds out the validation cells, runs the
generations and builds the result.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from GAMIPY.bio.fitness import TargetCells
from GAMIPY.config import GAConfig
from GAMIPY.data.data import Data
from GAMIPY.strategy_factory import StrategyFactory

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
    - **factory (StrategyFactory)**: Builds the operators below from `config`.
    - **fitness (ModelFitness)**, **selection (ColumnTournament)**, **crossover (ArithmeticModelCrossover)**,
      **mutation (GaussianModelMutation)**: The genetic operators, from the Bio module.
    - **data (Data)**: The dataset as given.
    - **targets (list)**: The columns with a missing value, whose models are evolved.
    - **generations_run (int)**: Generations evaluated by the last ``run()``.
    - **best_validation_error (float | None)**: Validation error of the returned models.
    - **history (list)**: One dict per generation of the last ``run()``: ``generation``,
      ``best_training_error`` (lowest total training error in that generation) and
      ``best_validation_error`` (mean over targets of the best validation error so far; NaN
      without validation cells).
    """

    def __init__(self, config: GAConfig, data: Data, factory: StrategyFactory | None = None) -> None:
        self.config = config
        self.factory = factory if factory is not None else StrategyFactory(config)
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
        self.fitness = self.factory.model_fitness(
            targets=self.targets, numeric=self.numeric, feature_slots=self.feature_slots,
            output_slots=self.output_slots, chain_rounds=CHAIN_ROUNDS,
        )
        self.selection = self.factory.model_selection()
        self.crossover = self.factory.model_crossover()
        self.mutation = self.factory.model_mutation()

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

    # --- data --------------------------------------------------------------

    def _cells(self, frame: pd.DataFrame) -> TargetCells:
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

    def _next_generation(
        self, weights: FloatArray, bias: FloatArray, errors: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """
        The next population. The fitness is a sum over target columns and each
        target has its own model, so selection works per target (cooperative
        coevolution): a child's model for a target comes from tournaments on
        that target's error (``ColumnTournament``), so good models for
        different columns found in different individuals are combined. Kept as they are: the individuals
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
                a, b = self.selection.select(filled[:, t]), self.selection.select(filled[:, t])
                w[slots], c[slots] = self.crossover.cross(weights[a, slots], bias[a, slots],
                                                          weights[b, slots], bias[b, slots])
            children_w[i], children_b[i] = self.mutation.mutate(w, c, self.allowed)
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
        validation: TargetCells = {}
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
            errors, judged = self.fitness.evaluate(weights, bias, features, missing, training, validation)
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
        outputs = self.fitness.predict(best[1][None], best[2][None], features, missing)[0]
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
