"""Threads for the NumPy parts of a GA run.

Scoring a population is the same computation for every individual (or
model), on NumPy arrays: matrix products, reductions, moments. NumPy releases
Python's global interpreter lock (GIL) inside these operations, so splitting
the population into chunks and scoring them in threads uses several cores.

Only this deterministic scoring is threaded. Everything that draws random
numbers (initialization, selection, crossover, mutation) stays in the calling
thread, in the same order, so a run gives the same result for any number of
threads. Python-level work (building individuals and genes) holds the GIL and
would not get faster in threads.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from typing import TypeVar

import numpy as np

T = TypeVar('T')

# Threads used when ``n_jobs`` is None. Measured on 8 cores: scoring gets
# faster up to about 4 threads and slower beyond (the chunks get too small
# for the threads to pay off).
AUTO_THREADS = 4
# Fewer items than this per chunk aren't worth a thread.
MIN_CHUNK = 16


def thread_count(n_jobs: int | None) -> int:
    """Threads for `n_jobs`: None -> ``min(AUTO_THREADS, cores)``, -1 -> all
    cores, a positive number -> that many.

    Raises
    ------
    - **ValueError**: If `n_jobs` is 0 or below -1.
    """
    cores = os.cpu_count() or 1
    if n_jobs is None:
        return max(1, min(AUTO_THREADS, cores))
    if n_jobs == -1:
        return cores
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be None, -1 or a positive integer, got {n_jobs!r}.")
    return n_jobs


def chunks(total: int, threads: int) -> list[slice]:
    """`total` items split into at most `threads` contiguous slices of at
    least ``MIN_CHUNK`` items (one slice when there are too few)."""
    parts = max(1, min(threads, total // MIN_CHUNK))
    edges = np.linspace(0, total, parts + 1).astype(int)
    return [slice(int(a), int(b)) for a, b in pairwise(edges.tolist())]


def map_chunks(function: Callable[[slice], T], total: int, threads: int) -> list[T]:
    """`function` applied to each slice of ``chunks(total, threads)``, in
    threads when there is more than one slice; results in slice order."""
    slices = chunks(total, threads)
    if len(slices) == 1:
        return [function(slices[0])]
    with ThreadPoolExecutor(max_workers=len(slices)) as pool:
        return list(pool.map(function, slices))
