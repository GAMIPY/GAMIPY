# All package metadata lives in pyproject.toml; this shim only keeps legacy
# `python setup.py ...` / editable-install tooling working.
from setuptools import setup

setup()
