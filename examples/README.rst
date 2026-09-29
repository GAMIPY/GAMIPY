Examples
========

Run from the repository root (with GAMIPY installed, e.g. ``pip install -e .``):

``quickstart.py``
    The smallest complete run: impute a small numerical + categorical
    DataFrame, print it before and after, and list the filled cells.

``evaluate_imputation.py``
    Hides 10% of the known values in a sample of the bundled ``student``
    dataset (numerical and categorical columns), imputes them, and scores the
    result against the hidden truth: RMSE / MAE / NRMSE for numerical columns,
    accuracy for categorical ones, next to a mean/mode baseline. The scoring
    comes from ``GAMIPY.evaluation`` (``column_errors`` /
    ``imputation_scores``), which you can use on your own data.

``compare_strategies.py``
    Runs several initialization x selection x crossover combinations on the
    same problem (iris) and ranks them by error.

The GA settings are kept small so each script finishes in seconds; increase
``population_size`` and ``generations`` for real use.
