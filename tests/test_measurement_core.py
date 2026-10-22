import numpy as np
import pytest

import app
from app import euclidean_distance, fit_board_grid, measure_grid, polygon_area, square_index


def test_distance_area_and_square_id_calculations():
    assert euclidean_distance((0, 0), (3, 4)) == 5
    assert polygon_area([(0, 0), (4, 0), (3, 2), (0, 2)]) == 7
    assert square_index(1, 2) == "R2C3"


def test_detected_grid_marks_only_interior_cells_as_direct():
    points = np.array(
        [(100 + 20 * column, 80 + 25 * row) for row in range(3) for column in range(4)],
        dtype=float,
    )
    detection = {
        "success": True,
        "points": points,
        "pattern_size": (4, 3),
        "image_size": (300, 250),
    }
    grid, direct = fit_board_grid(detection)
    measurements = measure_grid(grid, direct)

    assert len(measurements) == 20
    assert (measurements["measurement_source"] == "directly detected").sum() == 6
    assert (measurements["measurement_source"] == "extrapolated boundary estimate").sum() == 14


def test_grid_fit_rejects_boundary_extrapolations_outside_source_frame():
    points = np.array([(20 + 10 * x, 20 + 10 * y) for y in range(2) for x in range(2)])
    detection = {
        "success": True,
        "points": points,
        "pattern_size": (2, 2),
        "image_size": (35, 35),
    }
    with pytest.raises(ValueError, match="outside the original frame"):
        fit_board_grid(detection)


def test_sb_subpixel_corners_are_not_refined_a_second_time(monkeypatch):
    points = np.array([[40.25, 50.75], [80.5, 50.5], [40.5, 90.25], [80.75, 90.5]])
    monkeypatch.setattr(
        app,
        "_detect_on",
        lambda gray, pattern: (points.astype(np.float32).reshape(-1, 1, 2), "findChessboardCornersSB"),
    )

    def unexpected_refinement(*args, **kwargs):
        raise AssertionError("SB corners must not receive a second generic refinement.")

    monkeypatch.setattr(app.cv2, "cornerSubPix", unexpected_refinement)
    image = np.zeros((150, 150, 3), dtype=np.uint8)
    detection = app.find_checkerboard(image, 3, 3, (0, 0, 150, 150))

    assert detection["success"]
    np.testing.assert_allclose(detection["points"], points)
