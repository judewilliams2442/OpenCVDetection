import numpy as np
import pytest

from app import convert_preview_coordinates, manual_board_grid, measure_grid, roi_from_bounds


def test_coordinate_conversion_rejects_malformed_or_nonfinite_points():
    with pytest.raises(ValueError):
        convert_preview_coordinates(np.array([[np.nan, 1]]), 10, 10, 20, 20)
    with pytest.raises(ValueError):
        convert_preview_coordinates(np.array([[1, 2, 3]]), 10, 10, 20, 20)


def test_manual_grid_rejects_degenerate_or_malformed_corners():
    with pytest.raises(ValueError):
        manual_board_grid([(0, 0), (1, 1), (2, 2), (3, 3)], 3, 3)
    with pytest.raises(ValueError):
        manual_board_grid([(0, 0), (1, 0), (1, 1)], 3, 3)


def test_grid_measurement_requires_finite_coordinates_and_matching_mask():
    grid = np.array([[[0, 0], [1, 0]], [[0, 1], [1, 1]]], dtype=float)
    with pytest.raises(ValueError):
        measure_grid(grid, np.zeros((3, 3), dtype=bool))
    grid[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        measure_grid(grid)


def test_roi_bounds_convert_to_origin_and_size():
    assert roi_from_bounds(12, 112, 24, 74, 200, 100) == (12, 24, 100, 50)


@pytest.mark.parametrize(
    "bounds",
    [
        (-1, 20, 0, 10, 100, 100),
        (0, 101, 0, 10, 100, 100),
        (10, 10, 0, 10, 100, 100),
        (0, 20, 20, 19, 100, 100),
        (0, 20, 0, 10, 100.0, 100),
    ],
)
def test_roi_bounds_reject_invalid_ranges(bounds):
    with pytest.raises(ValueError):
        roi_from_bounds(*bounds)
