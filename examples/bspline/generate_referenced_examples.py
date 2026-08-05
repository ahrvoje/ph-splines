"""Render PHBSpline versions of every referenced input-gallery case.

Run from the repository root:
    python examples/bspline/generate_referenced_examples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_EXAMPLES_ROOT = ROOT / "examples" / "cubic"
sys.path.insert(0, str(SOURCE_EXAMPLES_ROOT))
sys.path.insert(0, str(ROOT))

import generate_examples
import generate_near_break
import generate_nonconvex_pathological
import generate_nonconvex_shapes

from examples.bspline._common import render_curve
from ph_spline import PHBSpline

OUT = Path(__file__).resolve().parent


def collections():
    yield "base", generate_examples.EXAMPLES
    yield "nonconvex", generate_nonconvex_shapes.EXAMPLES
    yield (
        "pathological",
        [case[:3] for case in generate_nonconvex_pathological.EXAMPLES],
    )
    yield "near_break", generate_near_break.CASES


def main() -> None:
    failures = []
    rendered = 0
    for group, cases in collections():
        for index, (name, points, note) in enumerate(cases, 1):
            try:
                curve = PHBSpline(points)
                render_curve(
                    curve,
                    OUT / group / f"{index:02d}_{name}.png",
                    f"{group} {index:02d} · {name}",
                    note,
                )
                rendered += 1
                print(f"ok {group:12s} {index:02d} {name}")
            except Exception as exc:  # noqa: BLE001 - complete corpus audit
                failures.append((group, index, name, exc))
                print(f"FAIL {group} {index:02d} {name}: {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit(
            f"{len(failures)} of {rendered + len(failures)} examples failed"
        )
    print(f"Rendered all {rendered} referenced input cases to {OUT}")


if __name__ == "__main__":
    main()
