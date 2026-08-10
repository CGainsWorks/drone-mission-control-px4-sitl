#!/usr/bin/env python3
"""Run the MPC-versus-PID comparison and write ``results/runs.csv``.

Every quantitative claim in the README comes out of this file, via
``analysis/make_results.py`` and ``analysis/make_figures.py``. Nothing is
transcribed by hand.

Run:
    python analysis/run_comparison.py            # the full grid
    python analysis/run_comparison.py --quick    # one seed, calm air only
    python analysis/run_comparison.py -j 4       # cap the worker count
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# Before numpy, and before anything that imports it: one BLAS thread per
# process, with multiprocessing supplying the parallelism. See
# mission_control.parallel for why.
from mission_control.parallel import limit_blas_threads  # noqa: E402

limit_blas_threads()

from mission_control.analysis.comparison import run_comparison  # noqa: E402
from mission_control.tuning import tuning_source  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=os.path.join(ROOT, "results"),
                        help="directory for runs.csv and the per-run logs")
    parser.add_argument("--quick", action="store_true",
                        help="one seed, nominal speed, calm air only")
    parser.add_argument("-j", "--workers", type=int, default=None,
                        help="parallel workers (default: cores - 1)")
    parser.add_argument("--no-logs", action="store_true",
                        help="skip writing the per-run CSV and ULog files")
    args = parser.parse_args()

    print(f"gains: {tuning_source()}")
    rows = run_comparison(
        args.output,
        quick=args.quick,
        workers=args.workers,
        save_logs=not args.no_logs,
    )

    diverged = [row for row in rows if row.get("diverged")]
    if diverged:
        print(f"\n{len(diverged)} run(s) diverged:")
        for row in diverged:
            print(f"  {row['mission']} {row['speed']}/{row['wind']}/{row['corner_mode']} "
                  f"s{row['seed']} {row['tracker']}: {row['final_state']}")
        # A divergence is a finding, not a crash: it is reported in the results
        # rather than hidden, and it is not an error exit unless everything went
        # wrong.
        if len(diverged) == len(rows):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
