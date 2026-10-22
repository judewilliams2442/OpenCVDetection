import json

import cv2
import numpy as np

from app import (
    convert_preview_coordinates,
    csv_bytes,
    euclidean_distance,
    find_checkerboard,
    fit_board_grid,
    json_bytes,
    manual_board_grid,
    measure_grid,
    polygon_area,
    square_index,
)


def test_euclidean_distance():
    assert euclidean_distance((0, 0), (3, 4)) == 5


def test_quadrilateral_area_uses_polygon_area():
    assert polygon_area([(0, 0), (4, 0), (3, 2), (0, 2)]) == 7


def test_square_index_is_one_based():
    assert square_index(0, 0) == "R1C1"
    assert square_index(2, 3) == "R3C4"


def test_preview_coordinate_scaling_restores_original_coordinates():
    points = np.array([[10.0, 20.0], [30.0, 40.0]])
    restored = convert_preview_coordinates(points, 4, 3, 100, 50)
    np.testing.assert_allclose(restored, [[140, 110], [220, 170]])


def test_perspective_synthetic_grid_measurement_matches_known_projection():
    columns, rows = 4, 3
    corners = [(80, 60), (440, 80), (410, 350), (55, 320)]
    grid = manual_board_grid(corners, columns, rows)
    result = measure_grid(grid, manual=True)
    assert len(result) == columns * rows
    assert result.iloc[0]["square_id"] == "R1C1"
    assert (result["measurement_source"] == "manual perspective estimate").all()

    expected_tl = np.array(corners[0], dtype=float)
    expected_tr = np.array(corners[0], dtype=float) + (np.array(corners[1]) - corners[0]) / columns
    np.testing.assert_allclose(
        result.iloc[0][["top_left_x_px", "top_left_y_px"]].to_numpy(dtype=float),
        expected_tl,
        atol=1e-4,
    )
    assert np.isclose(
        result.iloc[0]["top_edge_px"],
        np.linalg.norm(expected_tr - expected_tl),
        atol=1e-3,
    )


def test_synthetic_checkerboard_detection_with_perspective_distortion():
    square_columns, square_rows, square_size = 5, 4, 80
    board = np.full((square_rows * square_size, square_columns * square_size), 255, dtype=np.uint8)
    for row in range(square_rows):
        for column in range(square_columns):
            if (row + column) % 2:
                board[
                    row * square_size : (row + 1) * square_size,
                    column * square_size : (column + 1) * square_size,
                ] = 0

    margin = 40
    padded_board = np.full(
        (square_rows * square_size + 2 * margin, square_columns * square_size + 2 * margin),
        255,
        dtype=np.uint8,
    )
    padded_board[
        margin : margin + square_rows * square_size,
        margin : margin + square_columns * square_size,
    ] = board
    source = np.float32(
        [[margin, margin], [margin + square_columns * square_size, margin],
         [margin + square_columns * square_size, margin + square_rows * square_size],
         [margin, margin + square_rows * square_size]]
    )
    destination = np.float32([[100, 50], [450, 90], [430, 400], [80, 350]])
    transform = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(padded_board, transform, (540, 480), borderValue=127)
    rgb = cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)

    detection = find_checkerboard(rgb, square_columns, square_rows, (0, 0, 540, 480))
    grid, direct = fit_board_grid(detection)
    actual = measure_grid(grid, direct_intersections=direct)

    canonical = np.array(
        [(column, row) for row in range(square_rows + 1) for column in range(square_columns + 1)],
        dtype=np.float32,
    )
    known_grid = cv2.perspectiveTransform(
        (canonical * square_size + np.array([margin, margin])).reshape(-1, 1, 2), transform
    ).reshape(square_rows + 1, square_columns + 1, 2)
    expected = measure_grid(known_grid, manual=True)
    np.testing.assert_allclose(
        np.sort(actual["average_width_px"].to_numpy()),
        np.sort(expected["average_width_px"].to_numpy()),
        atol=2.0,
    )
    np.testing.assert_allclose(
        np.sort(actual["average_height_px"].to_numpy()),
        np.sort(expected["average_height_px"].to_numpy()),
        atol=2.0,
    )


def test_csv_and_json_exports_keep_measurements_and_metadata():
    grid = np.array([[[0, 0], [10, 0]], [[0, 10], [10, 10]]], dtype=float)
    measurements = measure_grid(grid, manual=True)
    csv = csv_bytes(measurements).decode("utf-8")
    assert "top_left_x_px" in csv
    assert "manual perspective estimate" in csv

    encoded = json_bytes((10, 10, 3), measurements, {"square_columns": 1})
    payload = json.loads(encoded)
    assert payload["image"] == {"width_px": 10, "height_px": 10}
    assert payload["detection_settings"]["square_columns"] == 1
    assert payload["measurement_results"][0]["area_px2"] == 100
