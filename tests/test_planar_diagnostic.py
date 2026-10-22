import cv2
import numpy as np
import pytest

import app
from planar_diagnostic import analyze_planar_deviation


def _detection(points, rows=5, columns=6):
    return {
        "success": True,
        "points": np.asarray(points, dtype=float),
        "pattern_size": (columns, rows),
        "image_size": (800, 600),
        "method": "synthetic detector",
    }


def _projective_grid(rows=5, columns=6):
    ideal = np.array(
        [(column, row) for row in range(rows) for column in range(columns)],
        dtype=np.float32,
    )
    transform = np.array(
        [[32.0, 2.0, 120.0], [1.0, 27.0, 75.0], [0.003, 0.001, 1.0]],
        dtype=np.float64,
    )
    observed = cv2.perspectiveTransform(ideal.reshape(-1, 1, 2), transform).reshape(-1, 2)
    return ideal, observed


def test_perfect_projective_grid_has_near_zero_planar_residuals():
    _, points = _projective_grid()
    analysis = analyze_planar_deviation(_detection(points))

    assert analysis["metrics"]["corner_count"] == 30
    assert analysis["metrics"]["rmse_px"] < 1e-4
    assert analysis["metrics"]["maximum_px"] < 1e-4
    assert analysis["records"]["corner_id"].is_unique
    assert analysis["records"]["provenance"].eq("synthetic detector").all()


def test_localized_perturbation_is_reported_at_its_grid_location():
    ideal, points = _projective_grid()
    target = 2 * 6 + 4
    points[target] += [7.0, -5.0]
    analysis = analyze_planar_deviation(_detection(points))
    records = analysis["records"]

    maximum = records["error_magnitude_px"].idxmax()
    assert records.loc[maximum, "corner_id"] == "R3C5"
    assert records.loc[maximum, "grid_row"] == 3
    assert records.loc[maximum, "grid_column"] == 5
    assert ideal[target].tolist() == [4.0, 2.0]


def test_shared_grid_corners_are_listed_exactly_once():
    _, points = _projective_grid(rows=3, columns=4)
    analysis = analyze_planar_deviation(_detection(points, rows=3, columns=4))

    assert len(analysis["records"]) == 12
    assert analysis["records"]["corner_id"].nunique() == 12


def test_generated_boundary_points_are_not_included_in_fit_or_metrics():
    _, points = _projective_grid()
    detection = _detection(points)
    analysis = analyze_planar_deviation(detection)
    generated_grid, direct = app.fit_board_grid(detection)
    generated_boundary_points = generated_grid[~direct]

    assert len(analysis["records"]) == len(points)
    assert all(
        not np.any(np.all(analysis["observed_points"] == point, axis=1))
        for point in generated_boundary_points
    )
    np.testing.assert_array_equal(analysis["observed_points"], points)


def test_explicit_sparse_indices_keep_corner_ids_stable():
    _, points = _projective_grid(rows=3, columns=4)
    kept = np.array([0, 1, 2, 4, 6, 7, 8, 9, 10, 11])
    sparse_detection = _detection(points[kept], rows=3, columns=4)
    indices = np.array([(index // 4, index % 4) for index in kept], dtype=np.int64)

    analysis = analyze_planar_deviation(sparse_detection, grid_indices=indices)

    assert analysis["records"].loc[3, "corner_id"] == "R2C1"
    assert analysis["records"].loc[3, "grid_row"] == 2
    assert analysis["records"].loc[3, "grid_column"] == 1
    with pytest.raises(ValueError, match="explicit grid indices"):
        analyze_planar_deviation(sparse_detection)


def test_duplicate_indices_and_collinear_coverage_are_rejected():
    _, points = _projective_grid(rows=3, columns=3)
    detection = _detection(points, rows=3, columns=3)
    indices = np.array(
        [(0, 0), (0, 1), (1, 0), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    )
    with pytest.raises(ValueError, match="Duplicate checkerboard grid indices"):
        analyze_planar_deviation(detection, grid_indices=indices)

    line_points = np.array([[10, 20], [20, 20], [30, 20], [40, 20]], dtype=float)
    line_detection = _detection(line_points, rows=2, columns=2)
    with pytest.raises(ValueError, match="non-collinear"):
        analyze_planar_deviation(line_detection)


def test_detection_resize_and_crop_mapping_use_separate_axis_scales(monkeypatch):
    monkeypatch.setattr(app, "MAX_DETECTION_SIDE", 4)
    detected_preview_points = np.array([[0.5, 0.5], [2.5, 0.5], [0.5, 1.5], [2.5, 1.5]])
    def return_detected_corners(gray, pattern):
        assert gray.shape == (2, 4)
        assert pattern == (2, 2)
        return (
            detected_preview_points.astype(np.float32).reshape(-1, 1, 2),
            "findChessboardCornersSB",
        )

    monkeypatch.setattr(app, "_detect_on", return_detected_corners)
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    roi = (7, 9, 5, 3)
    result = app.find_checkerboard(image, 3, 3, roi)

    assert result["coordinate_transform"]["scale_x"] == pytest.approx(5 / 4)
    assert result["coordinate_transform"]["scale_y"] == pytest.approx(3 / 2)
    np.testing.assert_allclose(
        result["points"],
        detected_preview_points * [5 / 4, 3 / 2] + [7, 9],
    )


def test_ransac_is_additional_and_does_not_change_baseline_metrics():
    _, points = _projective_grid()
    points[12] += [20, -10]
    detection = _detection(points)
    baseline = analyze_planar_deviation(detection)
    compared = analyze_planar_deviation(detection, ransac_threshold_px=1.0)

    assert compared["metrics"] == baseline["metrics"]
    assert "ransac_classification" in compared["records"]
    assert len(compared["records"]) == len(points)
