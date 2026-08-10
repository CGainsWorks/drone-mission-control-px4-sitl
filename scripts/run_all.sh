#!/usr/bin/env bash
# The whole pipeline, and the entry point `docker compose up` runs.
#
# Everything the README claims comes out of this script, in this order:
#   1. the test suite
#   2. the comparison over the full scenario grid
#   3. every failsafe scenario, flown with a real fault injected
#   4. the figures
#   5. docs/RESULTS.md and docs/FAILSAFE_MATRIX.md
#   6. a verification pass that the artifacts exist and are non-empty
#
# It ends with END-TO-END PASSED, or a non-zero exit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

# Which interpreter to use. `python` on Debian-slim and on CI, `python3` on
# distributions that ship only the versioned name, and $PYTHON for anyone whose
# interpreter is somewhere else entirely — a Git Bash shell on Windows, for
# instance, where the Windows Python is not on the shell's PATH under any name.
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  else
    echo "no python interpreter found; set PYTHON=/path/to/python" >&2
    exit 127
  fi
fi

QUICK=""
SKIP_TESTS=""
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK="--quick" ;;
    --skip-tests) SKIP_TESTS="1" ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

step() { printf '\n=== %s\n' "$1"; }

if [ -z "$SKIP_TESTS" ]; then
  step "1/6  test suite"
  "$PYTHON" -m pytest -q
else
  step "1/6  test suite (skipped)"
fi

step "2/6  controller comparison"
"$PYTHON" analysis/run_comparison.py $QUICK

step "3/6  failsafe scenarios"
"$PYTHON" analysis/run_failsafe_matrix.py

step "4/6  figures"
"$PYTHON" analysis/make_figures.py

step "5/6  results and failsafe matrix"
"$PYTHON" analysis/make_results.py

step "6/6  verifying artifacts"
missing=0
check() {
  if [ ! -s "$1" ]; then
    echo "  MISSING or empty: $1"
    missing=1
  else
    echo "  ok: $1"
  fi
}

check results/runs.csv
check results/comparison_meta.json
check results/failsafe_runs.csv
check docs/RESULTS.md
check docs/FAILSAFE_MATRIX.md
check analysis/figures/rms_by_scenario.png
check analysis/figures/error_split.png
check analysis/figures/compute_cost.png
check analysis/figures/trajectories.png
check analysis/figures/estimator.png

# Every scenario must have been flown by both controllers, and each run must
# have produced a log. A silently truncated grid would still write a valid CSV.
"$PYTHON" - <<'PY'
import csv, json, os, sys

with open("results/runs.csv", newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

meta = json.load(open("results/comparison_meta.json", encoding="utf-8"))
problems = []

if len(rows) != meta["runs"]:
    problems.append(f"runs.csv has {len(rows)} rows, metadata says {meta['runs']}")

paired = {}
for row in rows:
    key = (row["mission"], row["speed"], row["wind"], row["corner_mode"], row["seed"])
    paired.setdefault(key, set()).add(row["tracker"])
unpaired = [key for key, trackers in paired.items() if trackers != {"pid", "mpc"}]
if unpaired:
    problems.append(f"{len(unpaired)} scenario(s) not flown by both controllers")

if os.path.isdir("results/logs"):
    logs = [name for name in os.listdir("results/logs") if name.endswith(".csv")]
    if len(logs) != len(rows):
        problems.append(f"{len(logs)} per-run logs for {len(rows)} runs")

diverged = [row for row in rows if row["diverged"] == "1"]
if diverged:
    print(f"  note: {len(diverged)} run(s) diverged; they are reported in RESULTS.md")

if problems:
    for problem in problems:
        print(f"  FAILED: {problem}")
    sys.exit(1)
print(f"  ok: {len(rows)} runs, {len(paired)} scenarios, all paired")
PY

if [ "$missing" -ne 0 ]; then
  echo
  echo "END-TO-END FAILED: expected artifacts are missing"
  exit 1
fi

echo
echo "END-TO-END PASSED"
