.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/hasanaliozkan-dev/GAMIPY/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

GAMIPY could always use more documentation, whether as part of the
official GAMIPY docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/hasanaliozkan-dev/GAMIPY/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `GAMIPY` for local development.

1. Fork the `GAMIPY` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/GAMIPY.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv GAMIPY
    $ cd GAMIPY/
    $ pip install -e . -r requirements_dev.txt
    $ pre-commit install

   ``pre-commit install`` makes ruff, mypy and a few file checks run on every
   commit (``.pre-commit-config.yaml``). The mypy hook needs `uv
   <https://docs.astral.sh/uv/>`_; without it, commit with ``SKIP=mypy``
   and let CI run mypy.

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the linters
   (ruff and mypy) and the tests, including testing other Python versions
   with tox::

    $ make lint
    $ make test
    Or
    $ make test-all

   All of these are installed by ``requirements_dev.txt``.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 3.10 through 3.13. The GitHub
   Actions CI (``.github/workflows/ci.yml``) runs the tests on every supported
   version plus ruff and mypy; make sure it passes on your pull request.

Coverage
--------

CI measures line + branch coverage on the Python 3.13 job, shows it in the
run's summary, uploads the HTML report as the ``coverage-html`` artifact, and
fails if it drops below ``fail_under`` in ``pyproject.toml``. Locally::

$ make coverage

To track the trend over time on codecov.io, sign in there with GitHub, enable
the repository, and add its upload token as a ``CODECOV_TOKEN`` repository
secret. Until then the upload step is skipped.

Tips
----

To run a subset of tests::

$ pytest tests.test_GAMIPY


Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst)
and CI is green on ``main``. Then run::

$ bump2version patch # possible: major / minor / patch
$ git push
$ git push --tags

``bump2version`` (configured in ``.bumpversion.cfg``) updates
``__version__`` in ``src/GAMIPY/__init__.py`` (the single source of the
version), commits, and creates a ``vX.Y.Z`` tag. Pushing the tag starts
``.github/workflows/release.yml``, which:

1. runs the full CI (tests on Python 3.10-3.13, ruff, mypy);
2. checks that the tag matches ``__version__``;
3. builds the sdist and wheel and runs ``twine check``;
4. publishes them to PyPI with Trusted Publishing.

Nothing is published if any step fails.

One-time setup, before the first release: on pypi.org, add a *pending
Trusted Publisher* for the ``GAMIPY`` project (repository ``GAMIPY/GAMIPY``,
workflow ``release.yml``, environment ``pypi``), and create a ``pypi``
environment in the GitHub repository settings (optionally requiring manual
approval). No API token is stored anywhere.

To publish by hand instead (e.g. to TestPyPI), run ``make release``.

Documentation is built by Read the Docs from ``.readthedocs.yaml``. One-time
setup: import the repository on readthedocs.org; after that every push to
``main`` rebuilds https://GAMIPY.readthedocs.io.

Code of Conduct
---------------

Please note that this project is released with a `Contributor Code of Conduct`_.
By participating in this project you agree to abide by its terms.

.. _`Contributor Code of Conduct`: CODE_OF_CONDUCT.rst
