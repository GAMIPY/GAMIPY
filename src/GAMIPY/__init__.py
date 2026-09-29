
__author__ = """Hasan Ali Özkan"""
__email__ = 'hasanaliozkan@mu.edu.tr'
__version__ = '0.1.0'

from . import config, evaluation, imputation, runner, strategy_factory
from .bio import column_statistics, crossover, distance, encoding, fitness, mutation, selection, value_sampling
from .data import categorical, categorical_codec, column, data, missing_index, mtypes, numerical
from .datasets import sample_data
from .GAMIPY import GAMIPY
from .population import gene, individual, population
