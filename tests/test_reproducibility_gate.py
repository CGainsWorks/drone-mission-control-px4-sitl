"""The reproducibility gate has to keep failing on the things it exists for.

A gate that stops catching drift does not announce itself: it goes green, which
is exactly what it does when everything is fine. The previous version of this
check compared rendered PNGs and wall-clock timings across machines and so could
never pass; the risk when narrowing it is the opposite one, of excluding so much
that a real change slips through. These tests pin both edges.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from check_reproducible import BEGIN, END, strip_machine_dependent  # noqa: E402

DOCUMENT = f"""# Results

| Controller | RMS |
| --- | ---: |
| MPC | 0.7497 m |

## What it costs

{BEGIN}

| Controller | Step |
| --- | ---: |
| MPC | 1073.3 us |

{END}

## Estimator
"""


def _with(original: str, old: str, new: str) -> str:
    assert old in original, f"fixture no longer contains {old!r}"
    return original.replace(old, new)


class TestStripping:
    def test_marked_region_is_removed(self):
        stripped = strip_machine_dependent(DOCUMENT)
        assert "1073.3 us" not in stripped
        assert BEGIN not in stripped and END not in stripped

    def test_everything_outside_the_markers_survives(self):
        stripped = strip_machine_dependent(DOCUMENT)
        assert "0.7497 m" in stripped
        assert "## Estimator" in stripped
        assert "# Results" in stripped


class TestWhatTheGateCatches:
    def test_a_changed_timing_compares_equal(self):
        """The whole point: wall-clock numbers must not fail the build."""
        other = _with(DOCUMENT, "1073.3 us", "4462.2 us")
        assert strip_machine_dependent(DOCUMENT) == strip_machine_dependent(other)

    def test_a_changed_result_compares_different(self):
        """The other whole point: a moved result must fail the build."""
        other = _with(DOCUMENT, "0.7497 m", "0.9999 m")
        assert strip_machine_dependent(DOCUMENT) != strip_machine_dependent(other)

    def test_a_changed_heading_compares_different(self):
        other = _with(DOCUMENT, "## Estimator", "## Estimation")
        assert strip_machine_dependent(DOCUMENT) != strip_machine_dependent(other)


class TestMalformedMarkers:
    def test_an_unterminated_marker_does_not_silently_pass(self):
        """Dropping the tail is safe; comparing equal when content differs is not."""
        truncated = DOCUMENT.replace(END, "")
        other = _with(truncated, "0.7497 m", "0.9999 m")
        assert strip_machine_dependent(truncated) != strip_machine_dependent(other)

    def test_a_document_with_no_markers_is_unchanged_but_for_trailing_space(self):
        plain = "# Results\n\n| MPC | 0.7497 m |\n"
        assert strip_machine_dependent(plain) == plain


class TestGeneratorAgreement:
    def test_the_markers_match_the_generator(self):
        """Two files, one pair of strings. They drift silently if not pinned."""
        sys.path.insert(0, os.path.join(ROOT, "analysis"))
        import make_results

        assert make_results.NON_REPRODUCIBLE_BEGIN == BEGIN
        assert make_results.NON_REPRODUCIBLE_END == END

    def test_the_shipped_results_document_carries_the_markers(self):
        """If the generator stops emitting them, the gate starts comparing
        timings again and fails on every machine but one."""
        path = os.path.join(ROOT, "docs", "RESULTS.md")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        assert BEGIN in text and END in text
        assert "us |" not in strip_machine_dependent(text)
