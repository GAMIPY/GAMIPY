
import numpy as np
import pandas as pd


def _hidden_cells(original: pd.DataFrame, damaged: pd.DataFrame, imputed: pd.DataFrame) -> pd.DataFrame:
    """Returns the mask of cells known in `original` but missing in `damaged`.

    Raises
    ------
    - **ValueError**: If the three frames don't share the same columns and index, or `imputed` still
      has a missing value in one of those cells.
    """

    for name, frame in (("damaged", damaged), ("imputed", imputed)):
        if list(frame.columns) != list(original.columns) or not frame.index.equals(original.index):
            raise ValueError(f"`{name}` must have the same columns and index as `original`.")
    hidden = damaged.isna() & original.notna()
    unfilled = hidden & imputed.isna()
    if unfilled.any().any():
        cells = [(row, col) for col in unfilled.columns for row in unfilled.index[unfilled[col]]][:5]
        raise ValueError(f"`imputed` left {int(unfilled.to_numpy().sum())} hidden cell(s) missing, e.g. {cells}.")
    return hidden


def column_errors(original: pd.DataFrame, damaged: pd.DataFrame, imputed: pd.DataFrame) -> pd.DataFrame:
    """
    Per-column imputation error, measured only on the cells that are known in
    `original` but were hidden (missing) in `damaged`.

    Numerical columns get RMSE, MAE and NRMSE (RMSE divided by the column's
    range in `original`, so columns on different scales can be compared);
    categorical columns get accuracy.

    Args
    ----
    - **original (pd.DataFrame)**: The data with the true values.
    - **damaged (pd.DataFrame)**: `original` with some values hidden (set to NaN).
    - **imputed (pd.DataFrame)**: `damaged` after imputation.

    Returns
    -------
    - **pd.DataFrame**: One row per column that had hidden cells, indexed by column name, with
      `type`, `n`, `RMSE`, `MAE`, `NRMSE` and `accuracy`.

    Raises
    ------
    - **ValueError**: If the frames aren't aligned, or a hidden cell was left missing in `imputed`
      (it would otherwise silently drop out of the average).
    """

    hidden = _hidden_cells(original, damaged, imputed)
    rows = []
    for col in original.columns:
        mask = hidden[col]
        if not mask.any():
            continue
        truth, guess = original.loc[mask, col], imputed.loc[mask, col]
        if pd.api.types.is_numeric_dtype(original[col]):
            err = guess.astype(float) - truth.astype(float)
            rmse = float(np.sqrt((err ** 2).mean()))
            value_range = float(original[col].max() - original[col].min()) or 1.0
            rows.append({"column": col, "type": "numerical", "n": int(mask.sum()),
                         "RMSE": rmse, "MAE": float(err.abs().mean()),
                         "NRMSE": rmse / value_range, "accuracy": np.nan})
        else:
            rows.append({"column": col, "type": "categorical", "n": int(mask.sum()),
                         "RMSE": np.nan, "MAE": np.nan, "NRMSE": np.nan,
                         "accuracy": float((guess == truth).mean())})
    columns = ["type", "n", "RMSE", "MAE", "NRMSE", "accuracy"]
    if not rows:
        return pd.DataFrame(columns=columns).rename_axis("column")
    return pd.DataFrame(rows).set_index("column")[columns]


def imputation_scores(original: pd.DataFrame, damaged: pd.DataFrame, imputed: pd.DataFrame) -> dict[str, float]:
    """
    Overall imputation error on the hidden cells (see `column_errors`).

    Returns
    -------
    - **dict**: `n_numerical` / `n_categorical` (hidden cells scored), `RMSE` and `MAE` pooled over
      all hidden numerical cells, `NRMSE` averaged over numerical columns, and `accuracy` pooled over
      all hidden categorical cells (NaN when there are none of that kind).
    """

    errors = column_errors(original, damaged, imputed)
    numerical = errors[errors["type"] == "numerical"]
    categorical = errors[errors["type"] == "categorical"]
    n_num, n_cat = int(numerical["n"].sum()), int(categorical["n"].sum())
    return {
        "n_numerical": n_num,
        "n_categorical": n_cat,
        # Pool squared/absolute errors over cells, not columns.
        "RMSE": float(np.sqrt((numerical["RMSE"] ** 2 * numerical["n"]).sum() / n_num)) if n_num else np.nan,
        "MAE": float((numerical["MAE"] * numerical["n"]).sum() / n_num) if n_num else np.nan,
        "NRMSE": float(numerical["NRMSE"].mean()) if n_num else np.nan,
        "accuracy": float((categorical["accuracy"] * categorical["n"]).sum() / n_cat) if n_cat else np.nan,
    }


def distribution_scores(original: pd.DataFrame, damaged: pd.DataFrame, imputed: pd.DataFrame) -> dict[str, float]:
    """
    How well imputation preserves the distribution of the numerical columns,
    beyond cell-by-cell accuracy (mean imputation, for example, is accurate
    on average but shrinks every column's variance and weakens correlations).

    Returns
    -------
    - **dict**: ``variance_ratio``: mean over numerical columns with hidden cells of the imputed
      column's variance divided by the original column's (1 = preserved, below 1 = shrunk);
      ``correlation_error``: mean absolute difference between the off-diagonal correlations of the
      imputed and the original numerical columns; ``KS``: mean over numerical columns with hidden
      cells of the Kolmogorov-Smirnov statistic between the imputed and the true values of the hidden
      cells (0 = same distribution). NaN when not defined (e.g. no numerical column).

    Raises
    ------
    - **ValueError**: As ``column_errors``.
    """

    from scipy.stats import ks_2samp

    hidden = _hidden_cells(original, damaged, imputed)
    numeric = [col for col in original.columns if pd.api.types.is_numeric_dtype(original[col])]
    # The damaged data with only its hidden cells filled from `imputed`:
    # like column_errors, observed cells are never taken from `imputed`.
    completed = damaged[numeric].astype(float).mask(hidden[numeric], imputed[numeric].astype(float))
    truth_all = original[numeric].astype(float)
    ratios, ks = [], []
    for col in numeric:
        mask = hidden[col]
        if not mask.any():
            continue
        variance = float(np.nanvar(truth_all[col].to_numpy(dtype=float), ddof=1))
        if variance > 0:
            ratios.append(float(np.nanvar(completed[col].to_numpy(dtype=float), ddof=1)) / variance)
        ks.append(float(ks_2samp(completed.loc[mask, col], truth_all.loc[mask, col]).statistic))
    correlation_error = np.nan
    if len(numeric) >= 2:
        gap = (completed.corr() - truth_all.corr()).abs().to_numpy()
        off_diagonal = gap[~np.eye(len(numeric), dtype=bool)]
        off_diagonal = off_diagonal[np.isfinite(off_diagonal)]
        correlation_error = float(off_diagonal.mean()) if off_diagonal.size else np.nan
    return {
        "variance_ratio": float(np.mean(ratios)) if ratios else np.nan,
        "correlation_error": correlation_error,
        "KS": float(np.mean(ks)) if ks else np.nan,
    }
