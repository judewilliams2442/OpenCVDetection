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


def analyze_axis_nonlinearity(analysis: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Evaluate smooth X/Y residual trends with spatially blocked cross-validation."""
    records = analysis["records"]
    indices = np.asarray(analysis["grid_indices"], dtype=np.int64)
    image_width, image_height = analysis["image_size"]
    specifications = (
        ("x", 1, "observed_x_px", "dx_px", image_width),
        ("y", 0, "observed_y_px", "dy_px", image_height),
    )
    results: dict[str, dict[str, Any]] = {}

    for name, index_axis, coordinate_column, residual_column, image_extent in specifications:
        coordinate = (
            records[coordinate_column].to_numpy(dtype=np.float64) - image_extent / 2
        ) / (image_extent / 2)
        residual = records[residual_column].to_numpy(dtype=np.float64)
        axis_groups = indices[:, index_axis]
        unique_groups = np.unique(axis_groups)
        if len(unique_groups) < 5 or len(residual) < 12:
            results[name] = {
                "available": False,
                "interpretation": "Insufficient spatial coverage for blocked cross-validation.",
            }
            continue

        folds = np.array_split(unique_groups, min(5, len(unique_groups)))
        cv_errors: dict[int, float] = {}
        for degree in range(min(3, len(unique_groups) - 2) + 1):
            squared_errors = []
            for held_out_groups in folds:
                test = np.isin(axis_groups, held_out_groups)
                train = ~test
                if train.sum() <= degree:
                    continue
                design = np.vander(coordinate, N=degree + 1, increasing=True)
                coefficients, *_ = np.linalg.lstsq(design[train], residual[train], rcond=None)
                squared_errors.extend((design[test] @ coefficients - residual[test]) ** 2)
            if squared_errors:
                cv_errors[degree] = float(np.sqrt(np.mean(squared_errors)))

        if 0 not in cv_errors or 1 not in cv_errors:
            results[name] = {
                "available": False,
                "interpretation": "Insufficient independent spatial folds to compare trend models.",
            }
            continue

        best_degree = min(cv_errors, key=lambda degree: (cv_errors[degree], degree))
        best_rmse = cv_errors[best_degree]
        baseline_rmse = cv_errors[0]
        linear_rmse = cv_errors[1]
        improvement_over_constant = (
            0.0 if baseline_rmse == 0 else 1 - best_rmse / baseline_rmse
        )
        improvement_over_linear = (
            0.0 if linear_rmse == 0 else 1 - best_rmse / linear_rmse
        )
        if best_degree >= 2 and improvement_over_constant >= 0.1 and improvement_over_linear >= 0.1:
            interpretation = (
                f"A degree-{best_degree} nonlinear trend predicts held-out spatial bands "
                "better than both a constant and a linear model."
            )
        elif improvement_over_constant >= 0.1:
            interpretation = (
                "A smooth trend is present, but the data do not show a substantial "
                "cross-validated gain from nonlinear terms over a simpler model."
            )
        else:
            interpretation = (
                "No stable low-order spatial trend was detected; remaining variation may "
                "reflect local board shape, corner noise, lighting, or other effects."
            )

        design = np.vander(coordinate, N=best_degree + 1, increasing=True)
        coefficients, *_ = np.linalg.lstsq(design, residual, rcond=None)
        results[name] = {
            "available": True,
            "axis": name.upper(),
            "coordinate_column": coordinate_column,
            "residual_column": residual_column,
            "normalized_coordinates": coordinate,
            "residuals_px": residual,
            "best_degree": best_degree,
            "coefficients_px": coefficients,
            "cross_validated_rmse_px": cv_errors,
            "improvement_over_constant": improvement_over_constant,
            "improvement_over_linear": improvement_over_linear,
            "interpretation": interpretation,
            "fold_count": len(folds),
            "sample_count": len(residual),
        }

    return results
