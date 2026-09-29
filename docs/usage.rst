=====
Usage
=====

Quick start
-----------

Pass a ``pandas.DataFrame`` with missing values (``NaN``/``None``) to
:class:`~GAMIPY.GAMIPY.GAMIPY`, call ``run()``, and read the result from
``imputed_data``. Numerical and categorical columns are handled
automatically; categorical cells are always filled with one of the column's
real labels.

.. code-block:: python

    import numpy as np
    from GAMIPY import GAMIPY
    from GAMIPY.datasets.sample_data import SampleData

    df = SampleData().load_data("iris")

    # Knock out ~5% of the cells to have something to impute.
    rng = np.random.default_rng(0)
    damaged = df.mask(rng.random(df.shape) < 0.05)

    gamipy = GAMIPY(data=damaged, population_size=30, generations=10)
    gamipy.run()

    imputed = gamipy.imputed_data          # same shape and index as `damaged`
    assert not imputed.isnull().any().any()

``run()`` returns ``None``; the input DataFrame is never modified, and cells
that were observed in the input are left untouched in ``imputed_data``.

Choosing the genetic operators
------------------------------

Every operator is selected by name:

=========================== =============================================================================================================================================== ===========
Argument                    Accepted values                                                                                                                                 Default
=========================== =============================================================================================================================================== ===========
``initialization_strategy`` ``random``, ``heuristic``, ``hybrid``, ``zeros``, ``ones``, ``constant``                                                                        ``random``
``selection_method``        ``elitist``, ``tournament``, ``roulette``, ``rank``, ``steady_state``, ``stochastic_universal``                                                 ``elitist``
``crossover_type``          ``uniform``, ``one_point``, ``two_point``, ``arithmetic``, ``blend``                                                                            ``uniform``
``fitness_method``          ``cosine``, ``euclidean``, ``manhattan``, ``jaccard``, ``hamming``, ``chebyshev``, ``minkowski``, ``canberra``, ``braycurtis``, ``correlation`` ``cosine``
=========================== =============================================================================================================================================== ===========

An unknown name raises ``ValueError``. ``arithmetic`` and ``blend``
crossover average gene values, so they only accept data whose missing cells
are all numerical.

The remaining hyperparameters are ``population_size`` (default 100),
``generations`` (50), ``crossover_rate`` (0.1), ``mutation_rate`` (0.01),
``selection_number`` (10, must be at least 2), ``tournament_size`` (5, for
``tournament``) and ``ss_number`` (1, the number of individuals replaced per
generation by ``steady_state``).

.. code-block:: python

    gamipy = GAMIPY(
        data=damaged,
        population_size=30,
        generations=10,
        selection_method="tournament",
        tournament_size=4,
        crossover_type="two_point",
        crossover_rate=0.8,
        mutation_rate=0.1,
        fitness_method="euclidean",
    )
    gamipy.run()

Hyperparameters live on ``gamipy.config`` (a
:class:`~GAMIPY.config.GAConfig`), which can also be changed after
construction:

.. code-block:: python

    gamipy.config.generations = 5
    gamipy.config.selection_method = "rank"
    gamipy.run()

How the default GA avoids overfitting
-------------------------------------

GAMIPY is a pure genetic algorithm: no other imputation method seeds its
population, enters its fitness or replaces its result. Matching summary
statistics alone can be optimised past the truth, so by default the GA also:

* compares **standardized** statistics of the **whole** imputed dataset with
  the observed data (``standardize=True``, ``full_data_statistics=True``);
* mutates with small **Gaussian** steps (``mutation_type="gaussian"``);
* hides 10% of the observed cells as **validation cells**, returns the
  individual that imputes them best, and stops once that stops improving
  (``validation_fraction=0.1``, ``patience=10``). The hidden values are put
  back in ``imputed_data``.

``result_method="elite_mean"`` returns the average of the best individuals
instead of the single best. Every switch can be changed; for example, the
plain GA:

.. code-block:: python

    classic = GAMIPY(
        data=damaged, population_size=20, generations=5,
        validation_fraction=0, patience=0, standardize=False,
        full_data_statistics=False, mutation_type="uniform",
    )
    classic.run()
    print(classic.runner.generations_run)   # 5: no early stopping

Evolving prediction models instead of cell values
-------------------------------------------------

With ``representation="model"`` an individual is not one value per missing
cell but, for every column with a missing value, a linear model that
predicts it from the other columns of the row (one score per class for a
categorical column). The GA evolves these models by how well they predict the
observed cells; far fewer parameters than cells, so it overfits much less and
is the more accurate mode on the bundled benchmarks.

.. code-block:: python

    modelled = GAMIPY(data=damaged, population_size=20, generations=10, representation="model")
    modelled.run()
    assert not modelled.imputed_data.isnull().any().any()

Seeding with a heuristic solution
---------------------------------

The ``heuristic`` and ``hybrid`` strategies start from a guess you provide
(here: the middle of each column's observed range). ``heuristic_population`` has one row per
missing cell, ``(row, row_m, column, value)``, where ``row`` is the row label
in your DataFrame and ``row_m`` its position among the rows that contain a
missing value; :class:`~GAMIPY.data.missing_index.MissingIndexLocator`
produces the first three fields:

.. code-block:: python

    import pandas as pd
    from GAMIPY.data.data import Data
    from GAMIPY.data.missing_index import MissingIndexLocator

    numeric = damaged.drop(columns=["class"])
    seed = pd.DataFrame([
        (row, row_m, col, float((numeric[col].min() + numeric[col].max()) / 2))
        for row, row_m, col in MissingIndexLocator(Data(numeric)).locate()
    ])

    gamipy = GAMIPY(
        data=numeric,
        population_size=30,
        generations=10,
        initialization_strategy="hybrid",   # seeds + random individuals
        heuristic_population=seed,
    )
    gamipy.run()

Bundled datasets
----------------

:class:`~GAMIPY.datasets.sample_data.SampleData` ships the UCI datasets used
in the benchmarks: ``iris``, ``ecoli``, ``wine``, ``wdbc``, ``student`` and
``bank_marketing``.

.. code-block:: python

    wine = SampleData().load_data("wine")
    print(SampleData().data_info("iris")[:200])   # dataset description

Detecting the type of missingness
---------------------------------

:class:`~GAMIPY.data.mtypes.MissingnessTypes` estimates whether values are
missing completely at random (``MCAR``), at random (``MAR``) or not at
random (``MNAR``), as a diagnostic of your data:

.. code-block:: python

    from GAMIPY.data.mtypes import MissingnessTypes

    print(MissingnessTypes(Data(damaged)).m_type())   # 'MCAR', 'MAR' or 'MNAR'

MNAR data depends on the missing values themselves, which no method can
recover from the observed data alone.
