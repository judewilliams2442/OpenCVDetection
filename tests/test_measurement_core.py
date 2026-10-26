import numpy as np

import app
from planar_diagnostic import analyze_planar_deviation
from analysis_report import build_analysis_report_pdf
from app import (
    euclidean_distance,
    find_checkerboard_nearby_dimensions,
    fit_board_grid,
    measure_grid,
    polygon_area,
    square_index,
)


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


def test_grid_fit_measures_detected_cells_when_board_boundary_is_cropped():
    points = np.array([(20 + 10 * x, 20 + 10 * y) for y in range(2) for x in range(2)])
    detection = {
        "success": True,
        "points": points,
        "pattern_size": (2, 2),
        "image_size": (35, 35),
    }
    grid, direct = fit_board_grid(detection)

    assert detection["visible_only"]
    assert grid.shape == (2, 2, 2)
    assert direct.all()
    assert len(measure_grid(grid, direct)) == 1


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


def test_clipped_checkerboard_measures_only_squares_bounded_by_detected_corners():
    square_size = 60
    board_columns, board_rows = 28, 16
    board = np.empty(
        (board_rows * square_size, board_columns * square_size), dtype=np.uint8
    )
    for row in range(board_rows):
        for column in range(board_columns):
            board[
                row * square_size : (row + 1) * square_size,
                column * square_size : (column + 1) * square_size,
            ] = 255 if (row + column) % 2 == 0 else 0

    image = board[
        15 : 15 + 14 * square_size,
        20 : 20 + 26 * square_size,
    ]
    image_rgb = app.cv2.cvtColor(image, app.cv2.COLOR_GRAY2RGB)
    detection = app.find_checkerboard(
        image_rgb, 26, 14, (0, 0, image.shape[1], image.shape[0])
    )

    assert detection["success"]
    assert detection["visible_only"]
    assert detection["method"] == "findChessboardCornersSBWithMeta + CALIB_CB_LARGER"
    assert detection["detected_count"] == 25 * 13

    grid, direct = fit_board_grid(detection)
    measurements = measure_grid(grid, direct)
    assert grid.shape == (13, 25, 2)
    assert direct.all()
    assert len(measurements) == 24 * 12
    assert measurements["measurement_source"].eq("directly detected").all()
    assert app.annotate_image(image_rgb, grid, direct, measurements).shape == image_rgb.shape
    assert analyze_planar_deviation(detection)["metrics"]["corner_count"] == 25 * 13


def test_nearby_dimension_search_selects_largest_detected_subgrid(monkeypatch):
    def detect(image, columns, rows, roi):
        point_count = (columns - 1) * (rows - 1)
        if (columns, rows) not in {(25, 14), (26, 13)}:
            return {"success": False, "detected_count": 0}
        points = np.array(
            [(column * 10, row * 10) for row in range(rows - 1) for column in range(columns - 1)],
            dtype=float,
        )
        return {
            "success": True,
            "detected_count": point_count,
            "points": points,
            "pattern_size": (columns - 1, rows - 1),
        }

    monkeypatch.setattr(app, "find_checkerboard", detect)
    result = find_checkerboard_nearby_dimensions(
        np.zeros((10, 10, 3), dtype=np.uint8),
        26,
        14,
        (0, 0, 10, 10),
    )

    assert result["success"]
    assert result["requested_square_dimensions"] == (26, 14)
    assert result["detected_square_dimensions"] == (25, 14)
    assert result["dimension_adjustment"]
    assert result["visible_only"]


def test_pdf_report_includes_manual_measurements_without_claiming_residual_analysis():
    image = np.full((80, 100, 3), 220, dtype=np.uint8)
    grid = app.manual_board_grid(
        [(10, 10), (90, 12), (85, 70), (12, 68)],
        4,
        3,
    )
    direct = np.zeros(grid.shape[:2], dtype=bool)
    measurements = measure_grid(grid, direct, manual=True)
    detection = {
        "method": "manual four-corner perspective model",
        "square_columns": 4,
        "square_rows": 3,
        "detected_count": 0,
    }
    overlay = app.annotate_image(image, grid, direct, measurements)

    report = build_analysis_report_pdf(
        image,
        overlay,
        measurements,
        detection,
        (0, 0, 100, 80),
        "sample.png",
    )

    assert report.startswith(b"%PDF")
    assert len(report) > 5_000
    filtered_report = build_analysis_report_pdf(
        image,
        overlay,
        measurements,
        detection,
        (0, 0, 100, 80),
        "sample.png",
        include_estimates=False,
    )
    assert filtered_report.startswith(b"%PDF")
