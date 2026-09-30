# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Commit Conventions

Do not add `Co-Authored-By: Claude ...` or any other AI-attribution line to
git commit messages or pull request descriptions in this repository.

## Project Overview

**GAMIPY** is a Python library for missing data imputation using Genetic Algorithms (GAs), written as part of an M.Sc. thesis. Each missing cell is a gene; candidate imputations evolve towards the one whose statistical structure (means, skewness, correlations) best matches the observed data, with validation cells (hidden observed cells) guarding against overfitting. It handles numerical and categorical columns in the same DataFrame without manual preprocessing.

**GAMIPY must stay a pure GA** (the author's standing rule): no other imputer (mean, kNN, IterativeImputer, …) may seed the population, enter the fitness, replace the result, or be blended/selected by the GA. Other imputers belong only in benchmarks, as baselines; the benchmark and its baselines live in the separate development/research repository (GAMIPY_DEVELOPMENT).

Accuracy (benchmark of 6 datasets × 10%/30% MCAR × 3 masks × 3 GA seeds, NRMSE, lower is better): model representation (`representation='model'`, the default) 0.146, cells representation 0.386; baselines iterative imputer 0.146, kNN 0.175, mean 0.208. The benchmark, the measurements and the improvement plan live in GAMIPY_DEVELOPMENT. Don't state performance claims in docs without new measurements; `examples/compare_strategies.py` and `GAMIPY.evaluation` are the tools for measuring.

## Development Commands

Supported Python: **3.10–3.13**. Set up a development environment with:
```bash
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -e . -r requirements_dev.txt
```

**Testing:**
```bash
pytest                    # Run all tests (pytest.ini sets pythonpath = src)
make test-all            # Run tests via tox (py310–py313 + ruff + mypy envs)
make coverage            # Line+branch coverage; fails below fail_under in [tool.coverage] (pyproject.toml)
```

**Code Quality:** `make lint` runs `ruff check .` and `mypy`; both must stay clean.
- ruff (`ruff.toml`, pinned to 0.16.9 in `requirements_dev.txt`, tox and CI — keep the three in sync): pycodestyle errors/warnings incl. line length 120 (`E`, `W`, `E501`), pyflakes, import sorting (`I`), bugbear, pyupgrade (target py310), ruff's own rules. Nothing is ignored except `F401` in `src/GAMIPY/__init__.py` (re-exports). Wrap long code lines with `uvx ruff format --range N:N file` rather than formatting whole files.
- mypy runs in **strict** mode on `src/GAMIPY` (config in `pyproject.toml`), on Python 3.10 with `mypy==1.14.1` + `pandas-stubs==2.2.2.240807` (newer numpy stubs use 3.12-only syntax). Every function needs full annotations; missing cells are typed `list[MissingCell]` (`data/missing_index.py`). Without installing anything:
```bash
uv run --no-project --isolated --python 3.10 --with mypy==1.14.1 --with pandas-stubs==2.2.2.240807 --with-requirements requirements.txt mypy
```

`.pre-commit-config.yaml` runs ruff, mypy and file checks (YAML/TOML validity, merge markers, files over 500 KB) on commit, once `pre-commit install` has been run. Its mypy hook deliberately runs the exact CI toolchain through `uv` rather than mirrors-mypy, because mypy's results change with the numpy/pandas-stubs versions installed. Run everything with `uvx pre-commit run --all-files`.

**Docs:** published by Read the Docs from `.readthedocs.yaml` (Python 3.12, `pip install .` + `docs/requirements.txt`, whose pins must match `requirements_dev.txt`); theme is Furo, widened to the full page by `docs/_static/custom.css`. API pages are hand-written in `docs/source/` (no `sphinx-apidoc`): when adding a module, add its `.rst` there and list it under the matching "API reference" toctree in `docs/index.rst`. `docs/usage.rst` and `README.rst` Python blocks are executed by `tests/GAMIPY/test_docs_usage.py`, so keep them runnable. The docs build with zero warnings and CI's `docs` job enforces it (`sphinx-build -W`), so docstrings must be valid reStructuredText: section underlines at the header's own indentation and the same length, a blank line before a section header, and wrapped bullet lines indented two spaces under the `- `.

```bash
make docs                # Generate Sphinx documentation and open in browser
make servedocs           # Compile docs watching for changes (requires watchdog)
```

**Building & Distribution:**
```bash
make dist                # Build sdist + wheel (python -m build)
make release             # Manual upload to PyPI (normally done by release.yml)
make install             # Install package to active Python environment
make clean               # Remove all build, test, and Python artifacts
```

**Releasing:** the version lives only in `src/GAMIPY/__init__.py` (`pyproject.toml` reads it dynamically; `setup.py` is an empty shim). `bump2version patch` (`.bumpversion.cfg`) bumps, commits and tags `vX.Y.Z`; pushing the tag runs `.github/workflows/release.yml` (full CI as a gate → tag/version check → build + `twine check` → PyPI Trusted Publishing). The one-time PyPI/GitHub setup is described in `CONTRIBUTING.rst`. Bundled dataset CSVs and `.rst` descriptions are shipped via `[tool.setuptools.package-data]`.

**Other Python versions:** run the suite in a throwaway env with the unpinned `pyproject.toml` dependencies:
```bash
uv run --no-project --isolated --python 3.13 --with pandas --with numpy --with scipy --with scikit-learn --with pytest --with tqdm pytest -q -p no:cacheprovider
```

**Running specific tests** (tests are `unittest.TestCase` classes, run through pytest):
```bash
pytest tests/GAMIPY/data/test_data.py
pytest tests/GAMIPY/test_GAMIPY.py::TestRun::test_run_keeps_population_size_fixed_across_generations
```

## Architecture

The library is organized into four subpackages plus top-level orchestration modules:

### 1. **Data Module** (`src/GAMIPY/data/`)
- `data.py`: Main `Data` class, a thin DataFrame wrapper exposing column-type lists and the complete/missing row subsets
- `missing_index.py`: `MissingIndexLocator(data).locate()` returns one `MissingCell = (row, row_m, column)` tuple per missing cell, mapping the original row label to its position in the missing-rows subset
- `column.py`, `numerical.py`, `categorical.py`: column metadata/summaries (`CategoricalColumn`: `classes`, `n_classes`, `mode`)
- `categorical_codec.py`: `CategoricalCodec(classes)` - label ↔ integer-code mapping for one column
- `mtypes.py`: `MissingnessTypes.m_type()` classifies MCAR/MAR/MNAR by running `MissingnessTest` strategies (`LogisticRegressionMissingnessTest`, `ChiSquareMissingnessTest`) in order

### 2. **Population Module** (`src/GAMIPY/population/`)
- `gene.py`: `Gene(*, orig_location, m_location, name, data, dtype)` — all arguments required and keyword-only
- `individual.py`: `Individual` (one gene per missing cell). `gene_key()` is its identity for uniqueness; `_sampling_data(col)` samples from `m_data` when that column has ≥2 distinct observed values there, else from the full data (`fallback_data`)
- `population.py`: `Population` hierarchy; `initialize()` is a **method** (it mutates the population). Strategies: `random`; `heuristic` = the seed (`heuristic_population`) + variants of it (each resamples ~20% of the seed's genes, `variant_rate`); `hybrid` = half heuristic family, half random; `zeros`/`ones`/`constant`

### 3. **Bio Module** (`src/GAMIPY/bio/`)
- `crossover.py`: Uniform, OnePoint, TwoPoint, and the numeric-only Arithmetic/Blend (subclasses of `NumericCrossover`, which checks the two parents' dtypes)
- `mutation.py`: `Mutation.mutate` returns a new individual (never modifies its input); sampling is delegated to a `ValueSampler` (`value_sampling.py`)
- `selection.py`: 6 selection operators via `Selection.create(method, ...)` over `SELECTION_METHODS`; `select()` is a **method** (Elitist sorts the population in place; SteadyState returns a new list and leaves the population whole)
- `fitness.py`: `Fitness` scores individuals, using `encoding.CategoricalEncoder` and `distance` metrics; `_is_defined` drops statistics that are undefined for the data. `evaluate()` scores the whole population at once in NumPy (`_ImputedRows` stacks each individual's encoded missing rows; `_vectorized_scores`); `_calculate_score` / `_standardized_score` are the pandas reference implementations, still used for individuals the vectorized path can't handle (e.g. an empty cell) — `TestVectorizedFitness` keeps both within 1e-9
- `moments.py`: `Moments` — mean / sums of squared and cubed deviations / cross-products for a stack of datasets, `combine()` (Chan et al.) to add the complete rows' moments computed once; `skew()`/`cov()`/`corr()` follow pandas' conventions
- `distance.py`: `DistanceMetric` strategies in `DISTANCE_METRICS`, resolved by `DistanceMetric.create(name)`
- `column_statistics.py`, `value_sampling.py`: small strategies `Fitness`/`Mutation` depend on instead of the column classes

### 4. **Datasets Module** (`src/GAMIPY/datasets/`)
- `sample_data.py`: 6 bundled datasets (Bank Marketing, Ecoli, Iris, Student, WDBC, Wine); `load_data(name)` and `data_info(name)`

### 5. **Main API and orchestration** (`src/GAMIPY/*.py`)
`GAMIPY` (`GAMIPY.py`) is a thin user-facing facade; the work is split across:
- `config.py`: `GAConfig` dataclass holding every hyperparameter, plus the `selection_options` / `population_options` bags handed to the factories
- `strategy_factory.py`: `StrategyFactory(config)` maps config values onto the registry factories (`Population.create`, `Selection.create`, `Crossover.create`, `DistanceMetric.create`) and builds `Fitness`/`Mutation`; `population(..., strategy=, **option_overrides)` lets the runner override the configured strategy
- `runner.py`: `GeneticAlgorithmRunner(config, data, factory=None)` owns `m_data`/`c_data`/`missing_indices`/`population`/validation cells and the GA loop; `run()` returns the best `Individual`
- `imputation.py`: `ImputedDataBuilder(data).build(individual)` writes genes back onto a copy of the original DataFrame
- `model_search.py`: `ModelSearch(config, data)` for `representation='model'`: the GA evolves per-column linear prediction models (weights over the other columns, one score per class for categorical targets) instead of cell values; missing inputs are filled with the models' own predictions over `CHAIN_ROUNDS`; fitness = error on observed cells (cross-entropy for categorical targets), selection per target column (cooperative coevolution), small initial weights (`INIT_SCALE`); validation cells pick each target's model. Mean NRMSE 0.146 (= iterative imputer) vs 0.386 for the pure cells GA
- `parallel.py`: `thread_count(n_jobs)`, `chunks`, `map_chunks` — scores the population in threads over chunks of candidates (`ModelSearch._evaluate`; the moments in `Fitness._vectorized_scores`). Only deterministic NumPy work is threaded; RNG draws stay in the calling thread, so results are bit-identical for any `n_jobs` (`test_parallel.py`). Scores are computed from the joined moments, not per thread: BLAS `dot` rounding can depend on a vector's memory alignment, which gave 1-ulp differences. Measured gain (pop 200, 60 generations, 4 threads): model 1.04–1.37×, cells ~1.0× (cells time is Python `Gene` construction, GIL-bound). Parallel experiments should use processes with `n_jobs=1`
- `evaluation.py`: `column_errors` / `imputation_scores(original, damaged, imputed)` score an imputation on the hidden cells only (RMSE/MAE/NRMSE, categorical accuracy) and raise if a hidden cell is left unfilled; `distribution_scores` (variance ratio, correlation error, KS on the hidden cells) measures distribution preservation. Used by the examples and the docs

`GAMIPY.run()` returns `None` and stores the result on `gamipy.imputed_data`. Hyperparameters are on `gamipy.config`. Both searches (`gamipy.model` / `gamipy.runner`) keep `generations_run`, `best_validation_error` and `history` (one dict per generation: best training error or fitness score, best validation error so far).

**Adding a strategy:** subclass the ABC and add it to that module's registry dict (`POPULATION_STRATEGIES`, `SELECTION_METHODS`, `CROSSOVER_TYPES`, `DISTANCE_METRICS`). If the constructor needs extra arguments, declare them in the class-level `options` tuple (`Population`/`Selection`) and add the value to the matching `GAConfig` options bag. `GAMIPY` and the runner never import concrete classes. `test_integration.py` iterates the registries, so a new strategy is exercised end to end automatically.

**How the GA loop works (`GeneticAlgorithmRunner.run`):**
- `m_data` = rows with any missing value (index reset); `c_data` = complete rows. Fitness fills an individual's genes into `m_data` and compares it with `c_data`: `score = metric(means) + metric(skewness) + procrustes(covariances)`, all distances. **Lower score is better** — every selection method and `best_individual` minimise, and roulette/SUS weight individuals by `1 / (1 + score)`. `TestSelectionPrefersLowerScores` guards this: until 2026-09-28 selection maximised while `best_individual` minimised, so the GA evolved its worst solutions.
- Each generation: `Fitness.evaluate(rescore=False)` (survivors keep their score) → `selection.select()` → crossover on the top two selected → mutate the last selected → next generation = distinct survivors + offspring, topped up by `initialize_population(fill, exclude=...)`.
- **Validation cells** (`validation_fraction > 0`, the default): the runner hides that share of observed cells in `self.data` (`original_data` keeps the input), the GA imputes them too, every individual's error on them is tracked per generation (cached per individual, `_cached_validation_error`), the best one is returned (or the elite mean), and `patience` stops early. `build_result` puts the true values back. A user `heuristic_population` is re-aligned to `missing_indices` (`_aligned_seed`; uncovered cells get random values), since crossover pairs genes by position.
- `Fitness` gets data-derived context from `runner.fitness_context()`: a `CategoricalEncoder.for_data` built on the full data (one category list per column, so codes are comparable across datasets) and the observed-data reference for `full_data_statistics`.
- **Every individual in a population is distinct** (`gene_key()`): duplicates from selection with replacement, offspring equal to a parent, or deterministic strategies (zeros/ones/constant) are replaced by strategy-built, then random, individuals (up to `MAX_FILL_ROUNDS`). If the data can't supply enough distinct individuals, a `RuntimeWarning` is issued and duplicates keep the size fixed.
- Gene locations: `Gene._m_location` is a position in the reset-index `m_data`, while `Gene.location` is the original row label. `ImputedDataBuilder` remaps through the original missing-row order — don't use `_m_location[0]` as a label into the user's DataFrame.

## Configuration and Hyperparameters

`GAMIPY(...)` accepts these keyword arguments (stored on `GAConfig`):

- `population_size` (int, default 100), `generations` (int, default 50, must be ≥ 1)
- `crossover_rate` (float, default 0.1), `mutation_rate` (float, default 0.01)
- `selection_method` (str, default `'elitist'`): `SELECTION_METHODS` keys — note `'roulette'`, not `'roulette_wheel'`
- `selection_number` (int, default 10, must be ≥ 2 for crossover)
- `fitness_method` (str, default `'cosine'`): any `DISTANCE_METRICS` key except `'procrustes'` (reserved for the covariance term)
- `initialization_strategy` (str, default `'random'`): a `POPULATION_STRATEGIES` key
- `representation` (str, default `'model'`): `'model'` evolves per-column prediction models (`GAMIPY.model_search`), `'cells'` one gene per missing cell (`GeneticAlgorithmRunner`). The initialization, selection, crossover, fitness-metric and statistics options only apply to `'cells'`; tests of the cells GA's mechanics (`test_integration.py`, `TestRun`) set `representation='cells'` explicitly.
- `n_jobs` (int | None, default None = up to 4 threads; -1 all cores; 1 no threads): threads that score the population; doesn't change results
- Accuracy options: `validation_fraction` (0.1) + `patience` (10), `standardize` (True), `full_data_statistics` (True), `mutation_type` (`'gaussian'` or `'uniform'`) + `mutation_scale` (0.1), `result_method` (`'best'` / `'elite_mean'`) + `elite_size`. `validation_fraction=0, patience=0, standardize=False, full_data_statistics=False, mutation_type='uniform'` gives the classic GA; tests of classic GA mechanics set these explicitly.
- `crossover_type` (str, default `'uniform'`): `CROSSOVER_TYPES` keys; `arithmetic`/`blend` need all-numerical missing cells
- `tournament_size` (int, default 5), `ss_number` (int, default 1; `SteadyState.select()` returns `population_size - ss_number` individuals)
- `heuristic_population` (DataFrame, one `(row, row_m, column, value)` row per missing cell): required for `'heuristic'`/`'hybrid'`

An invalid name raises `ValueError` when that operator is first needed.

## Test Structure

Tests mirror the source code structure (`tests/GAMIPY/{data,bio,population,datasets}/`), plus:
- `test_GAMIPY.py` — the facade end to end; `test_runner.py` — config, factory, runner (incl. uniqueness), `ImputedDataBuilder`
- `test_integration.py` — every initialization × selection × crossover combination, every fitness metric, mixed categorical data, iris + MCAR. Slowest file (~10s); run it after touching the GA loop or any operator: several past breaks passed every unit test and only showed up here.
- `test_performance_options.py` (the GA's accuracy options, `Moments`, vectorized vs pandas fitness), `test_evaluation.py`, `test_examples.py`, `test_docs_usage.py`

Watch for duplicate test class names in one file: the second silently replaces the first (this hid 4 tests once). Tiny DataFrames can make covariance/skewness undefined inside `Fitness`, so `run()`-level tests use ~30-row random data. `pytest.ini` adds `src/` to the Python path. `SampleData.mock_data1()` is the source of truth for the mock dataset; `examples/mock_data.csv` is a dump of it, guarded by a test.

## Key Dependencies

**Runtime:** `pandas`, `numpy`, `scipy`, `scikit-learn` (unpinned in `pyproject.toml`; `requirements.txt` pins pandas 2.2.2 / scipy 1.13.0 / scikit-learn 1.5.0, tested by CI's pinned job).

**Development** (pinned in `requirements_dev.txt`; every pin must install on Python 3.10–3.13, see its header): `pytest` 9.1.1, `coverage` 7.16.2, `tox` 4.64.4, `ruff` 0.16.9, `mypy` 1.14.1 + `pandas-stubs`, `pre-commit`, `Sphinx` 8.1.3 + `furo`. `pip-audit` of this set was clean when pinned.

## Known Limitations

- **The cells GA overfits.** One gene per missing cell is hundreds to thousands of free parameters against ~50 validation cells, so the cells GA stays worse than mean imputation (0.386); the model representation (~columns² parameters) doesn't have this problem. Matching statistics (the original GAMIPY idea) didn't preserve distributions either: the classic GA inflates variance and has the worst downstream-task loss. The thesis experiments ran on pre-2026-09-28 code (everything maximised, so the covariance term rewarded mismatch).
- **The model representation is linear**: it misses non-linear relations (behind iterative on wdbc), and its categorical accuracy is still just below the majority class (0.606 vs 0.615).
- **Few missing rows weaken the GA.** With few missing rows, `m_data` statistics are noisy; skewness (needs ≥3 rows) and covariance (≥2) are left out of the fitness when undefined (`Fitness._is_defined`).
- Performance internals: `NumericalColumn` stats are lazy (`cached_property`), `Data` caches its column-type lists and per-column `column_profile` (what initialization samples from) until the DataFrame changes. Keep these caches in mind if you mutate a `Data`'s frame in place. `Gene` has `__slots__`; genes are shared read-only in practice — `Mutation`/crossover build new genes, never modify a parent's. The remaining run time is building `Gene` objects.
- Don't reference internal notes or TODO files from code comments or docstrings: docstrings are published in the API docs.

## CI

GitHub Actions (`.github/workflows/ci.yml`): tests on Python 3.10–3.13 with latest deps plus 3.10 with the `requirements.txt` pins, a lint job (ruff + mypy strict on 3.10), and a docs job (`sphinx-build -W`). The `mypy`/`ruff` versions there must match `requirements_dev.txt`. `release.yml` publishes tagged versions to PyPI; `security.yml` runs a weekly `pip-audit`; Dependabot keeps dependencies and actions current. `tox` is for local use only.
