"""Render PHBSplineOpen versions of every referenced input-gallery case.

Run from the repository root:
    python examples/bspline/generate_referenced_examples.py
"""

from __future__ import annotations

import math
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
from ph_spline import PHBSplineClosed, PHBSplineOpen

OUT = Path(__file__).resolve().parent


def closed_radial_star() -> list[list[float]]:
    """Return the exact 12-fold-symmetric nodes for the closed star case."""
    points = []
    for index in range(24):
        angle = math.pi * index / 12.0
        radius = 1.0 if index % 2 == 0 else 1.9
        points.append([radius * math.cos(angle), radius * math.sin(angle)])
    return points


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
                is_radial_star = name == "star_zigzag_radial"
                spline_type = PHBSplineClosed if is_radial_star else PHBSplineOpen
                curve = spline_type(
                    closed_radial_star() if is_radial_star else points,
                    g_order=8 if is_radial_star else 2,
                )
                render_curve(
                    curve,
                    OUT / group / f"{index:02d}_{name}.png",
                    f"{group} {index:02d} · {name}",
                    (
                        "closed 12-fold-symmetric radial star"
                        if is_radial_star
                        else note
                    ),
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
