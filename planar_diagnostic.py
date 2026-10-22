"""Planar-homography deviation analysis for detected checkerboard corners."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd


def analyze_planar_deviation(
    detection: dict[str, Any],
    grid_indices: Any | None = None,
    ransac_threshold_px: float | None = None,
) -> dict[str, Any]:
    """Fit an ideal unit grid to detected image points and report residuals.

    Grid indices are zero-based (row, column). Without explicit indices, only
    a complete row-major detector result is accepted.
    """
    if not detection.get("success"):
        raise ValueError("Cannot analyze an unsuccessful checkerboard detection.")
    points = np.asarray(detection.get("points"), dtype=np.float64)
    pattern = detection.get("pattern_size")
    image_size = detection.get("image_size")
    if (
        not isinstance(pattern, (tuple, list, np.ndarray))
        or np.asarray(pattern).shape != (2,)
        or any(not isinstance(value, (int, np.integer)) for value in pattern)
    ):
        raise ValueError("Checkerboard pattern dimensions are missing.")
    columns, rows = (int(value) for value in pattern)
    if columns < 2 or rows < 2:
        raise ValueError("At least a 2 by 2 internal-corner grid is required.")
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Detected coordinates must be finite x/y pairs.")
    if not image_size or len(image_size) != 2:
        raise ValueError("Original image dimensions are missing.")
    image_width, image_height = (int(value) for value in image_size)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Original image dimensions must be positive.")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Duplicate detected corner coordinates are not allowed.")

    if grid_indices is None:
        expected_count = rows * columns
        if len(points) != expected_count:
            raise ValueError(
                "A partial corner set requires explicit grid indices; corner IDs cannot be inferred."
            )
        indices = np.array(
            [(row, column) for row in range(rows) for column in range(columns)],
            dtype=np.int64,
        )
    else:
        indices = np.asarray(grid_indices)
        if indices.shape != (len(points), 2) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Grid indices must be integer (row, column) pairs for every point.")
        indices = indices.astype(np.int64, copy=False)
        if len(np.unique(indices, axis=0)) != len(indices):
            raise ValueError("Duplicate checkerboard grid indices are not allowed.")
        if (
            np.any(indices[:, 0] < 0)
            or np.any(indices[:, 0] >= rows)
            or np.any(indices[:, 1] < 0)
            or np.any(indices[:, 1] >= columns)
        ):
            raise ValueError("Grid indices fall outside the declared checkerboard pattern.")
    if len(points) < 4:
        raise ValueError("At least four detected corners are required to fit a homography.")

    ideal = indices[:, [1, 0]].astype(np.float32)
    if np.linalg.matrix_rank(ideal.astype(np.float64) - ideal.mean(axis=0)) < 2:
        raise ValueError("Detected grid indices do not provide non-collinear coverage.")
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise ValueError("Detected image points do not provide non-collinear coverage.")
    if cv2.contourArea(cv2.convexHull(points.astype(np.float32))) <= 0:
        raise ValueError("Detected image points do not enclose measurable spatial coverage.")

    matrix, _ = cv2.findHomography(ideal, points.astype(np.float32), 0)
    if matrix is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("The detected points do not produce a stable planar homography.")
    predicted = cv2.perspectiveTransform(
        ideal.reshape(-1, 1, 2), matrix
    ).reshape(-1, 2).astype(np.float64)
    if not np.isfinite(predicted).all():
        raise ValueError("The fitted homography produced non-finite corner predictions.")

    residuals = points - predicted
    magnitudes = np.linalg.norm(residuals, axis=1)
    records: dict[str, Any] = {
        "corner_id": [f"R{row + 1}C{column + 1}" for row, column in indices],
        "grid_row": indices[:, 0] + 1,
        "grid_column": indices[:, 1] + 1,
        "observed_x_px": points[:, 0],
        "observed_y_px": points[:, 1],
        "predicted_x_px": predicted[:, 0],
        "predicted_y_px": predicted[:, 1],
        "dx_px": residuals[:, 0],
        "dy_px": residuals[:, 1],
        "error_magnitude_px": magnitudes,
        "provenance": str(detection.get("method") or "detected checkerboard corner"),
    }

    ransac_mask: np.ndarray | None = None
    if ransac_threshold_px is not None:
        threshold = float(ransac_threshold_px)
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("The RANSAC threshold must be a positive finite pixel value.")
        if len(points) < 4:
            raise ValueError("At least four detected corners are required for RANSAC.")
        ransac_matrix, mask = cv2.findHomography(
            ideal, points.astype(np.float32), cv2.RANSAC, threshold
        )
        if ransac_matrix is None or mask is None:
            raise ValueError("The optional RANSAC comparison could not fit a homography.")
        ransac_mask = mask.reshape(-1).astype(bool)
        records["ransac_classification"] = np.where(ransac_mask, "inlier", "outlier")

    frame_area = image_width * image_height
    hull_area = float(cv2.contourArea(cv2.convexHull(points.astype(np.float32))))
    return {
        "records": pd.DataFrame(records),
        "homography": matrix,
        "predicted_points": predicted,
        "observed_points": points,
        "grid_indices": indices,
        "ransac_mask": ransac_mask,
        "image_size": (image_width, image_height),
        "metrics": {
            "corner_count": len(points),
            "coverage_area_px2": hull_area,
            "coverage_fraction": hull_area / frame_area,
            "rmse_px": float(np.sqrt(np.mean(magnitudes**2))),
            "median_px": float(np.median(magnitudes)),
            "p95_px": float(np.percentile(magnitudes, 95)),
            "maximum_px": float(np.max(magnitudes)),
        },
    }
