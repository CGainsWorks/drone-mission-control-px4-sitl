#!/usr/bin/env python3
"""Check that regenerating the committed documents changes nothing that should
not have changed.

Run it after `scripts/run_all.sh`, against a clean checkout. It compares what
is in the working tree now with what is committed at HEAD.

What is checked, and what is deliberately not
---------------------------------------------

The simulation is seeded and its outputs are sorted before writing, so every
number derived from a flight reproduces exactly — on any machine, in any order.
Those are worth gating on: if they move, either the committed documents are
stale or something has stopped being deterministic, and both matter.

Two things in the same directory do *not* reproduce, and gating on them only
produces a red build that says nothing:

* **The compute-cost timings.** They are wall-clock measurements of one machine
  under whatever else it was doing. Re-running locally on a busy machine moved
  them by 1.5x in the mean and 2.1x at the 99th percentile with every other
  number unchanged. `analysis/make_results.py` wraps them in
  `<!-- machine-dependent: ... -->` markers and this script strips the marked
  region from both sides before comparing.

* **The figures.** Matplotlib does not produce byte-identical PNGs across
  versions: the same data rendered on a GitHub runner differed from the local
  render by up to 17 kB, because font rasterisation and the bundled fonts
  differ. Comparing rendered images across machines is a test of the renderer,
  not of this repository. `scripts/run_all.sh` already checks that every figure
  was written and is non-empty, which is the part that can actually break.

This is the difference between a gate that has never passed and one that means
something. The workflow that ran before this script compared the PNG bytes, and
so could only ever have gone green on the one machine that generated them.
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Kept in step with analysis/make_results.py.
BEGIN = "<!-- machine-dependent: begin -->"
END = "<!-- machine-dependent: end -->"

#: Generated documents whose content must reproduce exactly, once the
#: machine-dependent regions are removed.
GATED = ("docs/RESULTS.md", "docs/FAILSAFE_MATRIX.md")


def strip_machine_dependent(text: str) -> str:
    """Drop every ``BEGIN``..``END`` region, inclusive.

    An unterminated ``BEGIN`` drops the rest of the document rather than
    raising: it is applied identically to both sides, so a malformed marker
    cannot make a comparison pass that should have failed.
    """
    out, depth = [], 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == BEGIN:
            depth += 1
            continue
        if stripped == END:
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            out.append(line)
    return "\n".join(out).strip() + "\n"


def committed(path: str) -> str | None:
    """The file as committed at HEAD, or None if it is not tracked."""
    # encoding is explicit, not `text=True`. Left to itself Python decodes a
    # subprocess with the locale encoding, which on a Windows checkout is
    # cp1252; these documents are full of em-dashes, so every one of them comes
    # back mangled and the comparison fails on a file that has not changed.
    result = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        cwd=ROOT, capture_output=True, encoding="utf-8",
    )
    return result.stdout if result.returncode == 0 else None


def main() -> int:
    failures = []
    for path in GATED:
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            failures.append(f"{path}: not generated")
            continue

        before = committed(path)
        if before is None:
            failures.append(f"{path}: not tracked, so there is nothing to compare")
            continue

        with open(full, encoding="utf-8") as handle:
            after = handle.read()

        if strip_machine_dependent(before) == strip_machine_dependent(after):
            print(f"  ok: {path}")
            continue

        failures.append(f"{path}: regenerating changed it")
        diff = subprocess.run(
            ["git", "diff", "--", path], cwd=ROOT,
            capture_output=True, encoding="utf-8", errors="replace",
        )
        print(f"\n--- {path} ---")
        print("\n".join(diff.stdout.splitlines()[:60]))

    if failures:
        print("\nreproducibility check FAILED:")
        for failure in failures:
            print(f"  {failure}")
        print(
            "\nEither the committed documents are stale — run "
            "`bash scripts/run_all.sh` and commit what it writes — or something "
            "that was deterministic no longer is. The timings and the figures "
            "are excluded from this check; see the docstring for why."
        )
        return 1

    print("\nreproducibility check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
