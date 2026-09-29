"""Smoke tests for examples/: the scripts must keep
running against the current API and actually produce/print a result."""
import contextlib
import io
import os
import sys
import unittest
import warnings

EXAMPLES = os.path.join(os.path.dirname(__file__), '..', '..', 'examples')


class TestExamples(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.abspath(EXAMPLES))

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(os.path.abspath(EXAMPLES))

    def run_main(self, module_name: str) -> str:
        module = __import__(module_name)
        out = io.StringIO()
        with warnings.catch_warnings(), contextlib.redirect_stdout(out):
            warnings.simplefilter('ignore')
            module.main()
        return out.getvalue()

    def test_quickstart_prints_the_imputed_data(self):
        output = self.run_main('quickstart')
        self.assertIn('After imputation', output)
        self.assertIn("filled 'D' in row 2", output)

    def test_evaluate_imputation_reports_errors_and_baseline(self):
        output = self.run_main('evaluate_imputation')
        self.assertIn('RMSE', output)
        self.assertIn('mean/mode', output)

    def test_compare_strategies_uses_valid_strategy_names(self):
        # Running it takes ~10 s; the combinations themselves are exercised
        # end to end by test_integration.py.
        from compare_strategies import COMBINATIONS

        from GAMIPY.bio.crossover import CROSSOVER_TYPES
        from GAMIPY.bio.selection import SELECTION_METHODS
        from GAMIPY.population.population import POPULATION_STRATEGIES
        for init, selection, crossover in COMBINATIONS:
            self.assertIn(init, POPULATION_STRATEGIES)
            self.assertIn(selection, SELECTION_METHODS)
            self.assertIn(crossover, CROSSOVER_TYPES)


if __name__ == '__main__':
    unittest.main()
