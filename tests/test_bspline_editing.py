"""Stable handles, snapshots, atomic edits, and edit transactions."""

from __future__ import annotations

import numpy as np
import pytest

from ph_spline import (
    DegeneratePointDataError,
    EditingPolicy,
    InvalidPointDataError,
    LocalEditFailure,
    NonFiniteCoordinateError,
    PHBSplineClosed,
    PHBSplineOpen,
    StaleHandleError,
    StaleLocationError,
)

POINTS = np.array([[0.0, 0.0], [1.0, 0.4], [2.0, -0.7], [3.0, 1.1], [2.2, 2.0]])


def test_move_preserves_handle_and_interpolates_new_position():
    curve = PHBSplineOpen(POINTS)
    handle = curve.point_handle(2)
    report = curve.move_point(handle, [2.1, -0.9], repair="global")
    assert curve.index_of(handle) == 2
    assert np.array_equal(curve.point(curve._knots[2]), [2.1, -0.9])
    assert report.version_before == 0 and report.version_after == 1


@pytest.mark.parametrize("order", [2, 4, 8])
def test_default_move_is_local_and_structurally_shares_exterior_spans(order):
    x = np.linspace(0.0, 20.0, 101)
    curve = PHBSplineOpen(np.column_stack((x, 0.2 * np.sin(x))), g_order=order)
    old_spans = curve._spans
    report = curve.move_point(50, [10.0, -0.08])
    assert report.rebuilt_span_count == 2 * (order + 3)
    affected = set(report.affected_span_ids)
    for index, span in enumerate(curve._spans):
        if index not in affected:
            assert span.preimage is old_spans[index].preimage
            assert span.position is old_spans[index].position
            assert span.arc is old_spans[index].arc


def test_default_insert_and_delete_are_local_and_keep_existing_handles():
    x = np.linspace(0.0, 20.0, 101)
    curve = PHBSplineOpen(np.column_stack((x, 0.2 * np.sin(x))))
    retained = curve.point_handle(70)
    inserted = curve.insert_point(50, [9.9, -0.05])
    assert inserted.report.rebuilt_span_count == 10
    assert curve.index_of(retained) == 71
    deleted = curve.delete_point(inserted.handle)
    assert deleted.rebuilt_span_count == 10
    assert curve.index_of(retained) == 70


@pytest.mark.parametrize("order", [2, 8])
def test_closed_seam_edits_use_a_wrapped_local_patch(order):
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    curve = PHBSplineClosed(points, g_order=order)
    report = curve.move_point(0, [1.02, 0.01])
    assert report.rebuilt_span_count == 2 * (order + 3)
    assert np.array_equal(curve.point(0.0), curve.point(1.0))


@pytest.mark.parametrize("index", [0, 100])
def test_open_endpoint_deletion_publishes_a_local_boundary_patch(index):
    x = np.linspace(0.0, 20.0, 101)
    points = np.column_stack((x, 0.2 * np.sin(x)))
    curve = PHBSplineOpen(points)
    report = curve.delete_point(index)
    assert report.rebuilt_span_count == 10
    expected = points[1:] if index == 0 else points[:-1]
    assert np.array_equal(curve.points, expected)


def test_strict_local_never_silently_escalates_to_global():
    x = np.linspace(0.0, 20.0, 101)
    curve = PHBSplineOpen(
        np.column_stack((x, 0.2 * np.sin(x))),
        editing=EditingPolicy(initial_patch_spans=2),
    )
    before = curve.points
    old_spans = curve._spans
    with pytest.raises(LocalEditFailure):
        curve.move_point(50, [10.0, -0.08])
    assert curve.version == 0
    assert np.array_equal(curve.points, before)
    assert curve._spans is old_spans


def test_expand_and_global_repair_are_explicit_in_reports():
    x = np.linspace(0.0, 20.0, 101)
    points = np.column_stack((x, 0.2 * np.sin(x)))
    editing = EditingPolicy(initial_patch_spans=2, max_patch_spans=16)
    expanded = PHBSplineOpen(points, editing=editing)
    report = expanded.move_point(50, [10.0, -0.08], repair="expand")
    assert report.rebuilt_span_count == 10
    global_curve = PHBSplineOpen(points, editing=editing)
    report = global_curve.move_point(50, [10.0, -0.08], repair="global")
    assert report.rebuilt_span_count == global_curve.num_spans


def test_insert_shifts_indices_without_changing_old_handles():
    curve = PHBSplineOpen(POINTS)
    old = curve.point_handle(2)
    result = curve.insert_point(2, [1.5, -0.1], repair="global")
    assert curve.index_of(result.handle) == 2
    assert curve.index_of(old) == 3
    assert curve.num_points == 6


def test_delete_stales_only_deleted_handle():
    curve = PHBSplineOpen(POINTS)
    deleted = curve.point_handle(2)
    retained = curve.point_handle(3)
    curve.delete_point(deleted, repair="global")
    with pytest.raises(StaleHandleError):
        curve.index_of(deleted)
    assert curve.index_of(retained) == 2


def test_append_prepend_and_list_insert_index_semantics():
    curve = PHBSplineOpen(POINTS)
    appended = curve.append_point([2.0, 3.0], repair="global")
    assert curve.index_of(appended.handle) == curve.num_points - 1
    prepended = curve.prepend_point([-1.0, -0.2], repair="global")
    assert curve.index_of(prepended.handle) == 0
    clamped = curve.insert_point(10_000, [3.0, 3.5], repair="global")
    assert curve.index_of(clamped.handle) == curve.num_points - 1


def test_failed_edit_is_exactly_atomic():
    curve = PHBSplineOpen(POINTS)
    before_points = curve.points
    before_version = curve.version
    before_handles = curve.point_handles
    with pytest.raises(DegeneratePointDataError):
        curve.move_point(1, POINTS[0], repair="global")
    assert curve.version == before_version
    assert curve.point_handles == before_handles
    assert np.array_equal(curve.points, before_points)


@pytest.mark.parametrize(
    "bad",
    [None, "xy", [1.0], [1.0, 2.0, 3.0], [True, 2.0], [1.0 + 2.0j, 2.0], [np.nan, 2.0]],
)
def test_malformed_edit_values_are_rejected_without_mutation(bad):
    curve = PHBSplineOpen(POINTS)
    before = curve.points
    spans = curve._spans
    with pytest.raises((InvalidPointDataError, NonFiniteCoordinateError)):
        curve.move_point(2, bad)
    assert curve.version == 0
    assert curve._spans is spans
    assert np.array_equal(curve.points, before)


def test_old_location_stales_after_commit():
    curve = PHBSplineOpen(POINTS)
    location = curve.location_at_length(0.4 * curve.length)
    curve.move_point(2, [2.1, -0.8], repair="global")
    with pytest.raises(StaleLocationError):
        curve.point(location)


def test_snapshot_remains_bitwise_stable_after_edits():
    curve = PHBSplineOpen(POINTS)
    snapshot = curve.snapshot(compact=True)
    samples = np.linspace(0.0, 1.0, 31)
    before = snapshot.points_at(samples)
    curve.move_point(2, [2.1, -0.8], repair="global")
    assert snapshot.version == 0
    assert np.array_equal(snapshot.points_at(samples), before)
    with pytest.raises(AttributeError):
        _ = snapshot.move_point


def test_transaction_commits_once_and_rolls_back_on_exception():
    curve = PHBSplineOpen(POINTS)
    with curve.edit(repair="global") as transaction:
        transaction.move_point(1, [1.1, 0.5])
        inserted = transaction.insert_point(2, [1.5, -0.1])
        transaction.delete_point(4)
    assert curve.version == 1
    assert curve.index_of(inserted) == 2
    before = curve.points
    with pytest.raises(RuntimeError), curve.edit(repair="global") as transaction:
        transaction.move_point(1, [9.0, 9.0])
        raise RuntimeError("abort")
    assert np.array_equal(curve.points, before)
