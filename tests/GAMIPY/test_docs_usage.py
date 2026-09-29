"""Runs every ``.. code-block:: python`` in docs/usage.rst and README.rst, in
order and in one shared namespace per file, so the usage guide and the README
quick start can't silently drift from the API (both used
to be cookiecutter stubs)."""
import contextlib
import io
import os
import re
import textwrap
import unittest
import warnings

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
USAGE_RST = os.path.join(ROOT, 'docs', 'usage.rst')
README_RST = os.path.join(ROOT, 'README.rst')


def run_python_blocks(test: unittest.TestCase, path: str) -> None:
    text = open(path, encoding='utf-8').read()
    blocks = re.findall(r'\.\. code-block:: python\n\n((?:    .*\n|\n)+)', text)
    test.assertGreater(len(blocks), 0)
    namespace: dict = {}
    name = os.path.basename(path)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        for i, block in enumerate(blocks, 1):
            with test.subTest(file=name, block=i):
                exec(compile(textwrap.dedent(block), f'{name} block {i}', 'exec'), namespace)


class TestUsageGuideExamples(unittest.TestCase):

    def test_all_python_examples_run(self):
        run_python_blocks(self, USAGE_RST)

    def test_readme_quick_start_runs(self):
        run_python_blocks(self, README_RST)


if __name__ == '__main__':
    unittest.main()
