"""Making multiprocessing and numpy coexist.

numpy and scipy ship with a multi-threaded BLAS which sizes its thread pool to
the core count by default. That is right for one process doing large linear
algebra and wrong for fifteen processes doing small linear algebra, which is
what the comparison harness and the tuning script do: fifteen workers each
spawning sixteen BLAS threads is 240 threads contending for 16 cores.

The largest matrices here are 15x15 in the estimator and 25x25 in the MPC's
condensed Hessian. Threading those was never going to pay — the crossover where
BLAS threading earns its overhead is a couple of orders of magnitude larger — so
the thread pool is capped at one per process and multiprocessing supplies the
parallelism instead.

The failure mode this avoids is worth knowing because it does not look like a
performance problem. Oversubscribed BLAS shows up as every worker sitting at low
CPU while the wall clock runs on, which reads as a deadlock rather than as
contention. If a parallel run ever looks hung, compare each worker's accumulated
CPU time against elapsed wall time before assuming it is stuck: near-100% means
it is simply working.

**This must run before numpy is imported.** The pool is sized at import time and
cannot be resized afterwards, so entry-point scripts call it as their first
statement, before any package import pulls numpy in. Spawned child processes
inherit the environment, so setting it in the parent covers them too.
"""

from __future__ import annotations

import os

#: The environment variables the various BLAS implementations read. Setting all
#: of them is the only portable way to cover OpenBLAS, MKL and Accelerate,
#: because which one numpy was built against is not knowable from here.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def limit_blas_threads(threads: int = 1) -> None:
    """Cap each process's BLAS thread pool. Call before importing numpy.

    Uses ``setdefault``, so an explicit setting in the environment always wins —
    somebody benchmarking on a machine where this is the wrong call should not
    have to edit the source to find out.
    """
    for variable in THREAD_VARIABLES:
        os.environ.setdefault(variable, str(threads))


def worker_count(requested: int = None) -> int:
    """How many worker processes to use.

    Leaves one core free. On a machine running this in the background that is
    the difference between a slow desktop and an unusable one, and the last
    worker buys very little once the others are saturated.
    """
    if requested is not None:
        return max(1, int(requested))
    return max(1, (os.cpu_count() or 2) - 1)
