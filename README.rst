======
GAMIPY
======


.. image:: https://img.shields.io/pypi/v/GAMIPY.svg
        :target: https://pypi.python.org/pypi/GAMIPY

.. image:: https://github.com/GAMIPY/GAMIPY/actions/workflows/ci.yml/badge.svg
        :target: https://github.com/GAMIPY/GAMIPY/actions/workflows/ci.yml

.. image:: https://readthedocs.org/projects/GAMIPY/badge/?version=latest
        :target: https://GAMIPY.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


GAMIPY imputes missing values in a pandas DataFrame with a genetic algorithm.
Each missing cell is a gene; candidate imputations are evolved towards the one
whose statistical structure (means, skewness and covariances) best matches the
complete rows of the dataset.

* Free software: GPL-3.0-or-later
* Documentation: https://GAMIPY.readthedocs.io
* Python 3.10 to 3.13


Installation
------------

.. code-block:: console

    $ pip install GAMIPY

or, from source::

    $ pip install git+https://github.com/GAMIPY/GAMIPY.git


Quick start
-----------

.. code-block:: python

    from GAMIPY import GAMIPY
    from GAMIPY.datasets.sample_data import SampleData

    data = SampleData.mock_data1()        # numerical + categorical, with NaNs

    gamipy = GAMIPY(data=data, population_size=30, generations=15)
    gamipy.run()
    print(gamipy.imputed_data)            # same shape and index, no NaNs left

See the `usage guide <https://GAMIPY.readthedocs.io/en/latest/usage.html>`_ and
the ``examples/`` directory, which also shows how to measure imputation error
(RMSE / MAE / accuracy) against a mean/mode baseline on your own data.


Features
--------

* **Mixed data types**: numerical and categorical columns in the same
  DataFrame, with no manual encoding; categorical cells are always filled with
  one of the column's real labels.
* **Pure genetic algorithm**: no other imputation method seeds the
  population, enters the fitness or replaces the result.
* **Structure-preserving fitness**: a candidate imputation is scored by how
  well the statistics (means, skewness, correlations) of the imputed dataset
  match the observed data.
* **Model representation** (``representation="model"``): instead of one value
  per missing cell, the GA evolves a small linear prediction model for every
  incomplete column - far fewer parameters, and the most accurate mode on the
  bundled benchmarks.
* **Guarded against overfitting**: by default part of the observed cells is
  hidden for validation; the GA returns the candidate that imputes them best
  and stops early once that stops improving.
* **Initialization strategies**: ``random`` (default), ``heuristic`` (your
  own starting guess plus variants of it), ``hybrid`` (half heuristic, half
  random), ``zeros``, ``ones``, ``constant``; every individual in a
  population is distinct.
* **6 selection methods**: ``elitist``, ``tournament``, ``roulette``, ``rank``,
  ``steady_state``, ``stochastic_universal``.
* **5 crossover operators**: ``uniform``, ``one_point``, ``two_point``,
  ``arithmetic``, ``blend``.
* **10 distance metrics** for the fitness function: ``cosine``, ``euclidean``,
  ``manhattan``, ``jaccard``, ``hamming``, ``chebyshev``, ``minkowski``,
  ``canberra``, ``braycurtis``, ``correlation`` (Procrustes distance is used for
  the covariance matrices).
* **Type-aware mutation**: numerical genes are resampled within the column's
  observed range, categorical genes from its known labels.
* **Bundled benchmark datasets**: Iris, Ecoli, Wine, WDBC, Student and Bank
  Marketing, via ``SampleData().load_data(name)``.
* **Extensible**: every operator is looked up by name in a registry, so a new
  strategy is a subclass plus one registry entry.


License
-------

Copyright (C) 2025 Hasan Ali Özkan

GAMIPY is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version (GPL-3.0-or-later). See ``LICENSE`` for the full text.


Credits
-------

GAMIPY was developed by Hasan Ali Özkan as part of an M.Sc. thesis at Muğla
Sıtkı Koçman University, under the supervision of Assist. Prof. Zeynep Filiz
Eren.

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
