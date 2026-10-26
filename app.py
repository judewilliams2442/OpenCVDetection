"""Local checkerboard pixel measurement application."""

from __future__ import annotations

import hashlib
import json
import math
from io import BytesIO
from typing import Any

import cv2
import numpy as np
import pandas as pd

MAX_IMAGE_PIXELS = 100_000_000
MAX_DETECTION_SIDE = 2400


def load_image(image_bytes: bytes) -> np.ndarray:
    """Decode a supported image into an RGB array after checking its size."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in {"PNG", "JPEG", "TIFF"}:
                raise ValueError("Upload a PNG, JPEG, or TIFF image.")
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"Image dimensions {width}x{height} exceed the safe limit "
                    f"of {MAX_IMAGE_PIXELS:,} pixels."
                )
            image.seek(0)
            return np.asarray(image.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable PNG, JPEG, or TIFF image.") from exc


def convert_preview_coordinates(
    points: np.ndarray,
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
) -> np.ndarray:
    """Map preview points into the source frame using scale and ROI offset."""
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("Coordinate scale factors must be positive.")
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.shape[-1] != 2 or not np.isfinite(coordinates).all():
        raise ValueError("Coordinates must be finite x/y pairs.")
    scaled = coordinates.copy()
    scaled[..., 0] = scaled[..., 0] * scale_x + offset_x
    scaled[..., 1] = scaled[..., 1] * scale_y + offset_y
    return scaled


def euclidean_distance(point_a: Any, point_b: Any) -> float:
    a = np.asarray(point_a, dtype=np.float64)
    b = np.asarray(point_b, dtype=np.float64)
    if a.shape != (2,) or b.shape != (2,) or not np.isfinite([*a, *b]).all():
        raise ValueError("Distance endpoints must be finite x/y coordinates.")
    return float(np.linalg.norm(b - a))


def polygon_area(points: Any) -> float:
    vertices = np.asarray(points, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError("A polygon requires at least three x/y coordinates.")
    if not np.isfinite(vertices).all():
        raise ValueError("Polygon coordinates must be finite.")
    x, y = vertices[:, 0], vertices[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def square_index(row: int, column: int) -> str:
    if row < 0 or column < 0:
        raise ValueError("Square row and column must be non-negative.")
    return f"R{row + 1}C{column + 1}"


def _validate_dimensions(square_columns: int, square_rows: int) -> None:
    if (
        not isinstance(square_columns, (int, np.integer))
        or not isinstance(square_rows, (int, np.integer))
        or square_columns < 1
        or square_rows < 1
    ):
        raise ValueError("Square columns and rows must be positive integers.")


def _region(image: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[np.ndarray, int, int]:
    if image is None or image.ndim not in (2, 3) or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("The source image is empty or malformed.")
    x, y, width, height = (int(value) for value in roi)
    image_height, image_width = image.shape[:2]
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > image_width or y + height > image_height:
        raise ValueError("The region of interest is outside the source image.")
    return image[y : y + height, x : x + width], x, y


def roi_from_bounds(
    left: int,
    right: int,
    top: int,
    bottom: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Convert exclusive right/bottom image bounds to an (x, y, width, height) ROI."""
    values = (left, right, top, bottom, image_width, image_height)
    if any(not isinstance(value, (int, np.integer)) for value in values):
        raise ValueError("Image bounds must be integer pixel coordinates.")
    if (
        image_width <= 0
        or image_height <= 0
        or left < 0
        or top < 0
        or right > image_width
        or bottom > image_height
        or right <= left
        or bottom <= top
    ):
        raise ValueError("Crop bounds must define a non-empty region inside the image.")
    return (left, top, right - left, bottom - top)


def _detect_on(gray: np.ndarray, pattern: tuple[int, int]) -> tuple[np.ndarray | None, str | None]:
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    try:
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
    except (cv2.error, AttributeError):
        found, corners = False, None
    if found and corners is not None:
        return corners, "findChessboardCornersSB"

    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    try:
        found, corners = cv2.findChessboardCorners(gray, pattern, flags=classic_flags)
    except cv2.error:
        found, corners = False, None
    if found and corners is not None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        return corners, "findChessboardCorners + cornerSubPix"
    return None, None


def find_checkerboard(
    image_rgb: np.ndarray,
    square_columns: int,
    square_rows: int,
    roi: tuple[int, int, int, int],
) -> dict[str, Any]:
    """Detect internal intersections and return their original-image coordinates."""
    _validate_dimensions(square_columns, square_rows)
    if square_columns < 3 or square_rows < 3:
        raise ValueError("Automatic detection needs at least 3 squares per side.")
    region, offset_x, offset_y = _region(image_rgb, roi)
    pattern = (square_columns - 1, square_rows - 1)
    scale = min(1.0, MAX_DETECTION_SIDE / max(region.shape[:2]))
    original_gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY) if region.ndim == 3 else region
    detection_gray = (
        cv2.resize(original_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else original_gray
    )
    scale_x = region.shape[1] / detection_gray.shape[1]
    scale_y = region.shape[0] / detection_gray.shape[0]
    enhanced_gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(detection_gray)
    for candidate_gray in (detection_gray, enhanced_gray):
        corners, method = _detect_on(candidate_gray, pattern)
        if corners is None:
            continue

        local_points = corners.reshape(-1, 2).astype(np.float64)
        local_points[:, 0] *= scale_x
        local_points[:, 1] *= scale_y
        # SB already returns subpixel corner positions. A second generic
        # refinement can pull them toward seams or uneven square edges.
        if method == "findChessboardCorners + cornerSubPix" and scale < 1.0:
            local_points[:, 0] = np.clip(local_points[:, 0], 0, region.shape[1] - 1)
            local_points[:, 1] = np.clip(local_points[:, 1], 0, region.shape[0] - 1)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            points_for_refine = local_points.astype(np.float32).reshape(-1, 1, 2)
            try:
                points_for_refine = cv2.cornerSubPix(
                    original_gray, points_for_refine, (5, 5), (-1, -1), criteria
                )
                local_points = points_for_refine.reshape(-1, 2).astype(np.float64)
            except cv2.error:
                pass
        points = local_points + np.array([offset_x, offset_y], dtype=np.float64)
        expected = pattern[0] * pattern[1]
        return {
            "success": len(points) == expected,
            "points": points,
            "pattern_size": pattern,
            "expected_count": expected,
            "detected_count": len(points),
            "method": method,
            "roi": tuple(int(value) for value in roi),
            "image_size": (int(image_rgb.shape[1]), int(image_rgb.shape[0])),
            "coordinate_transform": {
                "scale_x": scale_x,
                "scale_y": scale_y,
                "offset_x": offset_x,
                "offset_y": offset_y,
                "detection_width": int(detection_gray.shape[1]),
                "detection_height": int(detection_gray.shape[0]),
            },
            "failure_reason": None if len(points) == expected else "Detected intersection count is incomplete.",
        }
    return {
        "success": False,
        "points": np.empty((0, 2), dtype=np.float64),
        "pattern_size": pattern,
        "expected_count": pattern[0] * pattern[1],
        "detected_count": 0,
        "method": None,
        "roi": tuple(int(value) for value in roi),
        "image_size": (int(image_rgb.shape[1]), int(image_rgb.shape[0])),
        "coordinate_transform": {
            "scale_x": scale_x,
            "scale_y": scale_y,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "detection_width": int(detection_gray.shape[1]),
            "detection_height": int(detection_gray.shape[0]),
        },
        "failure_reason": (
            "No complete checkerboard was found. Adjust the ROI and grid dimensions, "
            "or use the four-corner manual estimate."
        ),
    }


def estimate_grid_dimensions(
    image_rgb: np.ndarray,
    roi: tuple[int, int, int, int],
    max_squares: int = 13,
) -> list[tuple[int, int]]:
    """Try plausible square counts on a reduced ROI; results must be user-confirmed."""
    region, _, _ = _region(image_rgb, roi)
    if max_squares < 3:
        raise ValueError("Maximum square count must be at least 3.")
    scale = min(1.0, 1200 / max(region.shape[:2]))
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY) if region.ndim == 3 else region
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    aspect = region.shape[1] / region.shape[0]
    candidates: list[tuple[float, int, int]] = []
    for columns in range(3, max_squares + 1):
        for rows in range(3, max_squares + 1):
            ratio = columns / rows
            if abs(math.log(ratio / aspect)) > 1.3:
                continue
            corners, _ = _detect_on(enhanced, (columns - 1, rows - 1))
            if corners is not None and len(corners) == (columns - 1) * (rows - 1):
                candidates.append((abs(math.log(ratio / aspect)), columns, rows))
    candidates.sort()
    return [(columns, rows) for _, columns, rows in candidates]


def fit_board_grid(detection: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Fit a planar grid, preserving detected intersections and extrapolating borders."""
    if not detection.get("success"):
        raise ValueError("Cannot fit a grid to a failed detection.")
    points = np.asarray(detection.get("points"), dtype=np.float64)
    columns, rows = (int(value) for value in detection["pattern_size"])
    if points.shape != (columns * rows, 2) or not np.isfinite(points).all():
        raise ValueError("Detected coordinates are missing or malformed.")
    canonical = np.array(
        [(column + 1, row + 1) for row in range(rows) for column in range(columns)],
        dtype=np.float32,
    )
    matrix, inliers = cv2.findHomography(canonical, points.astype(np.float32), 0)
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        raise ValueError("The detected points do not produce a stable homography.")
    if abs(np.linalg.det(matrix)) < 1e-12 or np.linalg.cond(matrix) > 1e12:
        raise ValueError("The homography is singular or unstable; inspect the ROI and detection.")
    square_columns, square_rows = columns + 1, rows + 1
    full_coordinates = np.array(
        [
            (column, row)
            for row in range(square_rows + 1)
            for column in range(square_columns + 1)
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    grid = cv2.perspectiveTransform(full_coordinates, matrix).reshape(
        square_rows + 1, square_columns + 1, 2
    ).astype(np.float64)
    if not np.isfinite(grid).all():
        raise ValueError("The fitted grid contains invalid extrapolated coordinates.")
    image_size = detection.get("image_size")
    if image_size:
        image_width, image_height = image_size
        outside = (
            (grid[:, :, 0] < 0)
            | (grid[:, :, 0] >= image_width)
            | (grid[:, :, 1] < 0)
            | (grid[:, :, 1] >= image_height)
        )
        invalid_cells = []
        for row in range(square_rows):
            for column in range(square_columns):
                if outside[row : row + 2, column : column + 2].any():
                    invalid_cells.append(square_index(row, column))
        if invalid_cells:
            raise ValueError(
                "Extrapolated board boundaries place these squares outside the original frame "
                f"({', '.join(invalid_cells)}); the board may be cropped, so automatic measurements were withheld."
            )
    projected = cv2.perspectiveTransform(canonical.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    residuals = np.linalg.norm(projected - points, axis=1)
    detection["reprojection_rms_px"] = float(np.sqrt(np.mean(residuals**2)))
    detection["reprojection_max_px"] = float(np.max(residuals))
    grid[1:-1, 1:-1] = points.reshape(rows, columns, 2)
    direct = np.zeros((square_rows + 1, square_columns + 1), dtype=bool)
    direct[1:-1, 1:-1] = True
    detection["homography"] = matrix.tolist()
    return grid, direct


def manual_board_grid(
    corners: Any, square_columns: int, square_rows: int
) -> np.ndarray:
    """Bilinearly interpolate an estimated grid from corners ordered TL, TR, BR, BL."""
    _validate_dimensions(square_columns, square_rows)
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Enter four finite corners in TL, TR, BR, BL order.")
    if cv2.contourArea(points.reshape(-1, 1, 2)) <= 1:
        raise ValueError("The outer board corners must form a non-degenerate polygon.")
    top_left, top_right, bottom_right, bottom_left = points.astype(np.float64)
    grid = np.empty((square_rows + 1, square_columns + 1, 2), dtype=np.float64)
    for row in range(square_rows + 1):
        v = row / square_rows
        for column in range(square_columns + 1):
            u = column / square_columns
            grid[row, column] = (
                (1 - u) * (1 - v) * top_left
                + u * (1 - v) * top_right
                + u * v * bottom_right
                + (1 - u) * v * bottom_left
            )
    if not np.isfinite(grid).all() or np.any(grid[:, :, 0] < 0) or np.any(grid[:, :, 1] < 0):
        raise ValueError("The projected manual grid has invalid image coordinates.")
    return grid


def measure_grid(
    grid: np.ndarray,
    direct_intersections: np.ndarray | None = None,
    manual: bool = False,
) -> pd.DataFrame:
    """Measure each cell using projected coordinates in the original frame."""
    vertices = np.asarray(grid, dtype=np.float64)
    if vertices.ndim != 3 or vertices.shape[2] != 2 or min(vertices.shape[:2]) < 2:
        raise ValueError("Grid must contain at least one cell and finite x/y coordinates.")
    if not np.isfinite(vertices).all():
        raise ValueError("Grid coordinates must be finite.")
    direct = (
        np.zeros(vertices.shape[:2], dtype=bool)
        if direct_intersections is None
        else np.asarray(direct_intersections, dtype=bool)
    )
    if direct.shape != vertices.shape[:2]:
        raise ValueError("Direct-intersection mask does not match the grid.")
    measurements = []
    for row in range(vertices.shape[0] - 1):
        for column in range(vertices.shape[1] - 1):
            tl, tr = vertices[row, column], vertices[row, column + 1]
            bl, br = vertices[row + 1, column], vertices[row + 1, column + 1]
            top = euclidean_distance(tl, tr)
            bottom = euclidean_distance(bl, br)
            left = euclidean_distance(tl, bl)
            right = euclidean_distance(tr, br)
            all_direct = bool(
                direct[row : row + 2, column : column + 2].all()
            )
            source = (
                "manual perspective estimate"
                if manual
                else "directly detected"
                if all_direct
                else "extrapolated boundary estimate"
            )
            record: dict[str, Any] = {
                "square_id": square_index(row, column),
                "row": row + 1,
                "column": column + 1,
                "top_left_x_px": float(tl[0]),
                "top_left_y_px": float(tl[1]),
                "top_right_x_px": float(tr[0]),
                "top_right_y_px": float(tr[1]),
                "bottom_left_x_px": float(bl[0]),
                "bottom_left_y_px": float(bl[1]),
                "bottom_right_x_px": float(br[0]),
                "bottom_right_y_px": float(br[1]),
                "top_edge_px": top,
                "bottom_edge_px": bottom,
                "left_edge_px": left,
                "right_edge_px": right,
                "average_width_px": (top + bottom) / 2,
                "average_height_px": (left + right) / 2,
                "diagonal_tl_br_px": euclidean_distance(tl, br),
                "diagonal_tr_bl_px": euclidean_distance(tr, bl),
                "area_px2": polygon_area([tl, tr, br, bl]),
                "measurement_source": source,
            }
            measurements.append(record)
    return pd.DataFrame(measurements)


def annotate_image(
    image_rgb: np.ndarray,
    grid: np.ndarray,
    direct_intersections: np.ndarray,
    measurements: pd.DataFrame,
    show_labels: bool = True,
    suspect: bool = False,
) -> np.ndarray:
    """Draw the grid overlay on a copy of the original-resolution RGB image."""
    annotated = image_rgb.copy()
    direct = np.asarray(direct_intersections, dtype=bool)
    rows, columns = len(grid) - 1, len(grid[0]) - 1
    for row in range(rows + 1):
        for column in range(columns + 1):
            point = tuple(np.rint(grid[row, column]).astype(int))
            for next_column in range(column + 1, columns + 1):
                endpoint = tuple(np.rint(grid[row, next_column]).astype(int))
                color = (30, 190, 90) if direct[row, column] and direct[row, next_column] else (255, 150, 30)
                cv2.line(annotated, point, endpoint, color, 2, cv2.LINE_AA)
            if row < rows:
                endpoint = tuple(np.rint(grid[row + 1, column]).astype(int))
                color = (30, 190, 90) if direct[row, column] and direct[row + 1, column] else (255, 150, 30)
                cv2.line(annotated, point, endpoint, color, 2, cv2.LINE_AA)
            cv2.circle(annotated, point, 5, (255, 40, 40) if direct[row, column] else (255, 170, 20), -1)
    for row in range(rows):
        for column in range(columns):
            corners = np.rint(
                [grid[row, column], grid[row, column + 1], grid[row + 1, column + 1], grid[row + 1, column]]
            ).astype(np.int32)
            cell_color = (
                (255, 220, 20)
                if suspect
                else (30, 190, 90)
                if direct[row : row + 2, column : column + 2].all()
                else (255, 150, 30)
            )
            cv2.polylines(annotated, [corners], True, cell_color, 2, cv2.LINE_AA)
            if show_labels:
                center = np.rint(corners.mean(axis=0)).astype(int)
                label = measurements.iloc[row * columns + column]["square_id"]
                cv2.putText(
                    annotated,
                    str(label),
                    tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.5, min(image_rgb.shape[:2]) / 1800),
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
    return annotated


def csv_bytes(measurements: pd.DataFrame) -> bytes:
    return measurements.to_csv(index=False, float_format="%.15g").encode("utf-8-sig")


def json_bytes(
    image_dimensions: tuple[int, int, int],
    measurements: pd.DataFrame,
    detection_settings: dict[str, Any],
) -> bytes:
    height, width = image_dimensions[:2]
    payload = {
        "image": {"width_px": int(width), "height_px": int(height)},
        "detection_settings": _json_safe(detection_settings),
        "measurement_results": _json_safe(measurements.to_dict(orient="records")),
    }
    return json.dumps(payload, indent=2, allow_nan=False).encode("utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Cannot export non-finite measurement values.")
    return value


def _png_bytes(image_rgb: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("OpenCV could not encode the annotated PNG.")
    return encoded.tobytes()


def _summary(measurements: pd.DataFrame, include_estimates: bool) -> pd.DataFrame:
    selected = measurements
    if not include_estimates:
        selected = measurements[measurements["measurement_source"] == "directly detected"]
    if selected.empty:
        return pd.DataFrame()
    columns = {
        "average_width_px": "Average horizontal dimension (px)",
        "average_height_px": "Average vertical dimension (px)",
        "area_px2": "Area (px²)",
    }
    return selected[list(columns)].rename(columns=columns).agg(
        ["mean", "median", "std", "min", "max"]
    )


def _planar_overlay(
    image_rgb: np.ndarray,
    analysis: dict[str, Any],
    grid: np.ndarray,
    direct_intersections: np.ndarray,
    arrow_scale: float,
) -> np.ndarray:
    """Draw detected and predicted points without hiding generated grid nodes."""
    annotated = image_rgb.copy()
    points = analysis["observed_points"]
    predicted = analysis["predicted_points"]
    hull = cv2.convexHull(np.rint(points).astype(np.int32))
    cv2.polylines(annotated, [hull], True, (255, 255, 0), 2, cv2.LINE_AA)

    for row, column in zip(*np.where(~np.asarray(direct_intersections, dtype=bool))):
        location = tuple(np.rint(grid[row, column]).astype(int))
        cv2.circle(annotated, location, 5, (255, 150, 20), 2, cv2.LINE_AA)
    for observed, fitted in zip(points, predicted):
        start = tuple(np.rint(observed).astype(int))
        end = tuple(np.rint(fitted).astype(int))
        cv2.circle(annotated, start, 5, (30, 220, 80), -1, cv2.LINE_AA)
        cv2.drawMarker(annotated, end, (40, 100, 255), cv2.MARKER_CROSS, 13, 2, cv2.LINE_AA)
        arrow_end = tuple(np.rint(observed + (fitted - observed) * arrow_scale).astype(int))
        cv2.arrowedLine(annotated, start, arrow_end, (220, 40, 220), 2, cv2.LINE_AA, tipLength=0.25)
    return annotated


def _show_planar_diagnostic(
    st: Any,
    image: np.ndarray,
    detection: dict[str, Any],
    grid: np.ndarray,
    direct: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    from ollama_summary import build_summary_context, summarize_with_ollama
    from planar_diagnostic import analyze_planar_deviation

    st.subheader("Deviation from planar homography")
    use_ransac = st.checkbox("Add separate RANSAC comparison", value=False, key="planar_use_ransac")
    threshold = (
        float(st.number_input(
            "RANSAC threshold (original-image pixels)",
            min_value=0.1,
            value=2.0,
            step=0.5,
            key="planar_ransac_threshold",
        ))
        if use_ransac
        else None
    )
    analysis = analyze_planar_deviation(detection, ransac_threshold_px=threshold)
    records = analysis["records"]
    metrics = analysis["metrics"]
    st.caption(
        f"{metrics['corner_count']} original detected corners; convex-hull coverage "
        f"{metrics['coverage_area_px2']:.1f} px² ({metrics['coverage_fraction']:.2%} of the frame). "
        "Only detected corners are fitted and scored; generated boundary points remain separate."
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("RMSE", f"{metrics['rmse_px']:.3f} px")
    metric_columns[1].metric("Median", f"{metrics['median_px']:.3f} px")
    metric_columns[2].metric("95th percentile", f"{metrics['p95_px']:.3f} px")
    metric_columns[3].metric("Maximum", f"{metrics['maximum_px']:.3f} px")

    arrow_scale = float(st.slider(
        "Residual arrow magnification",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=1.0,
        key="planar_arrow_scale",
    ))
    overlay = _planar_overlay(image, analysis, grid, direct, arrow_scale)
    preview_scale = min(1.0, 1400 / max(image.shape[:2]))
    if preview_scale < 1:
        overlay = cv2.resize(overlay, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA)
    st.image(
        overlay,
        caption=(
            "Green: observed; blue cross: predicted; magenta: residual arrow "
            f"({arrow_scale:g}×); orange rings: generated boundary; cyan: measured-region hull."
        ),
    )

    x = records["observed_x_px"].to_numpy()
    y = records["observed_y_px"].to_numpy()
    dx = records["dx_px"].to_numpy()
    dy = records["dy_px"].to_numpy()
    magnitude = records["error_magnitude_px"].to_numpy()
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].scatter(x, dx, c=dx, cmap="coolwarm")
    axes[0, 0].set(xlabel="Image x (px)", ylabel="dx (px)", title="dx vs image x")
    axes[0, 1].scatter(y, dx, c=dx, cmap="coolwarm")
    axes[0, 1].set(xlabel="Image y (px)", ylabel="dx (px)", title="dx vs image y")
    axes[1, 0].scatter(x, dy, c=dy, cmap="coolwarm")
    axes[1, 0].set(xlabel="Image x (px)", ylabel="dy (px)", title="dy vs image x")
    axes[1, 1].scatter(y, dy, c=dy, cmap="coolwarm")
    axes[1, 1].set(xlabel="Image y (px)", ylabel="dy (px)", title="dy vs image y")
    st.pyplot(figure, clear_figure=True)

    image_width, image_height = analysis["image_size"]
    figure, axis = plt.subplots(figsize=(8, 6))
    heatmap = axis.scatter(x, y, c=magnitude, cmap="viridis", s=55, marker="s")
    hull_points = cv2.convexHull(np.rint(analysis["observed_points"]).astype(np.int32)).reshape(-1, 2)
    closed_hull = np.vstack((hull_points, hull_points[:1]))
    axis.plot(closed_hull[:, 0], closed_hull[:, 1], color="cyan")
    axis.set(xlim=(0, image_width), ylim=(image_height, 0), xlabel="Image x (px)", ylabel="Image y (px)")
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("Residual magnitude (unmeasured regions blank)")
    figure.colorbar(heatmap, ax=axis, label="Residual magnitude (px)")
    st.pyplot(figure, clear_figure=True)

    row_summary = records.groupby("grid_row", as_index=False).agg(
        corner_count=("corner_id", "count"),
        mean_dx_px=("dx_px", "mean"),
        mean_dy_px=("dy_px", "mean"),
        mean_error_px=("error_magnitude_px", "mean"),
    )
    column_summary = records.groupby("grid_column", as_index=False).agg(
        corner_count=("corner_id", "count"),
        mean_dx_px=("dx_px", "mean"),
        mean_dy_px=("dy_px", "mean"),
        mean_error_px=("error_magnitude_px", "mean"),
    )
    left, right = st.columns(2)
    left.caption("Grid-row summaries")
    left.dataframe(row_summary.round(3), use_container_width=True)
    right.caption("Grid-column summaries")
    right.dataframe(column_summary.round(3), use_container_width=True)

    st.markdown("#### Optional local AI summary")
    st.caption(
        "Only computed metrics and corner summaries are sent to Ollama at localhost:11434; "
        "the image itself is not sent. AI text can be mistaken, so compare its statements with "
        "the measurements; it does not replace them."
    )
    model = st.text_input(
        "Ollama model",
        value="qwen3.5:4b",
        key="planar_ollama_model",
        help="Enter a model name installed in Ollama, such as qwen3.5:4b.",
    )
    summary_context = build_summary_context(analysis)
    summary_fingerprint = hashlib.sha256(
        json.dumps(summary_context, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    summary_key = f"planar_ollama_summary_{summary_fingerprint}_{model.strip()}"
    if st.button("Summarize locally with Ollama", key="planar_ollama_summarize"):
        st.session_state.pop(summary_key, None)
        try:
            with st.spinner(f"Waiting for local model {model.strip() or '(no model selected)'}..."):
                st.session_state[summary_key] = summarize_with_ollama(analysis, model)
        except (ValueError, RuntimeError) as exc:
            st.error(f"Local AI summary failed: {exc}")
    if summary_key in st.session_state:
        st.markdown(f"**Summary from `{model.strip()}`**")
        st.write(st.session_state[summary_key])

    st.dataframe(records.round(4), use_container_width=True, height=350)
    st.download_button(
        "Download planar deviation CSV",
        records.to_csv(index=False, float_format="%.15g").encode("utf-8-sig"),
        "planar_homography_deviation.csv",
        "text/csv",
    )


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Checkerboard Pixel Measurements", layout="wide")
    st.title("Checkerboard Pixel Measurement Tool")
    st.caption(
        "Measurements are reported in original-image pixels. Physical squares may have "
        "different projected pixel dimensions because of viewing angle, perspective, and lens distortion."
    )

    uploaded = st.file_uploader("Upload a PNG, JPEG, or TIFF frame", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if uploaded is None:
        st.info("Upload a frame to begin. Image processing stays local to this computer.")
        return
    image_bytes = uploaded.getvalue()
    image_signature = hashlib.sha256(image_bytes).hexdigest()
    if st.session_state.get("image_signature") != image_signature:
        for key in ("checkerboard_result", "manual_result", "grid_candidates", "accept_detection", "use_manual_result"):
            st.session_state.pop(key, None)
        st.session_state.pop("roi_horizontal_bounds", None)
        st.session_state.pop("roi_vertical_bounds", None)
        st.session_state["image_signature"] = image_signature
    try:
        image = load_image(image_bytes)
    except ValueError as exc:
        st.error(str(exc))
        return
    image_height, image_width = image.shape[:2]
    st.write(f"Original frame: **{image_width:,} × {image_height:,} px**")
    max_preview = 1400
    preview_scale = min(1.0, max_preview / max(image.shape[:2]))
    preview = cv2.resize(image, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA) if preview_scale < 1 else image
    st.subheader("1. Select the board region")
    st.caption(
        "Drag either handle on the horizontal or vertical range slider, or click a value to type an exact "
        "pixel bound. Right and bottom bounds are exclusive; the defaults select the full image."
    )
    horizontal, vertical = st.columns(2)
    left, right = horizontal.slider(
        "Left — right bounds (px)",
        min_value=0,
        max_value=image_width,
        value=(0, image_width),
        step=1,
        key="roi_horizontal_bounds",
    )
    top, bottom = vertical.slider(
        "Top — bottom bounds (px)",
        min_value=0,
        max_value=image_height,
        value=(0, image_height),
        step=1,
        key="roi_vertical_bounds",
    )
    try:
        roi = roi_from_bounds(left, right, top, bottom, image_width, image_height)
    except ValueError as exc:
        st.error(str(exc))
        return
    roi_preview = preview.copy()
    preview_x = int(round(left * preview_scale))
    preview_y = int(round(top * preview_scale))
    preview_right = int(round(right * preview_scale))
    preview_bottom = int(round(bottom * preview_scale))
    cv2.rectangle(roi_preview, (preview_x, preview_y), (preview_right, preview_bottom), (30, 120, 255), 3)
    st.image(roi_preview, caption="Original frame preview; selected crop is outlined in blue")

    st.subheader("2. Configure the checkerboard")
    st.caption(
        "Enter the total number of squares, not intersections. A board with C columns and R rows "
        "has (C−1) × (R−1) internal intersections."
    )
    dim_columns = st.columns(2)
    square_columns = int(dim_columns[0].number_input("Square columns", min_value=3, max_value=100, value=8))
    square_rows = int(dim_columns[1].number_input("Square rows", min_value=3, max_value=100, value=6))
    if st.button("Help estimate grid dimensions"):
        with st.spinner("Trying plausible checkerboard dimensions on the selected region..."):
            try:
                candidates = estimate_grid_dimensions(image, roi)
            except (ValueError, cv2.error) as exc:
                st.error(f"Grid estimation failed: {exc}")
            else:
                if candidates:
                    st.session_state["grid_candidates"] = candidates[:8]
                else:
                    st.warning("No plausible grid dimensions were detected. Set them manually.")
    if st.session_state.get("grid_candidates"):
        st.write("Candidate square columns × rows (confirm the matching dimensions above):")
        st.write(", ".join(f"{columns} × {rows}" for columns, rows in st.session_state["grid_candidates"]))

    st.subheader("3. Detect and verify")
    if st.button("Detect checkerboard", type="primary"):
        st.session_state.pop("accept_detection", None)
        st.session_state.pop("checkerboard_result", None)
        try:
            detection = find_checkerboard(image, square_columns, square_rows, roi)
            if detection["success"]:
                grid, direct = fit_board_grid(detection)
                detection["square_columns"] = square_columns
                detection["square_rows"] = square_rows
                detection["image_signature"] = image_signature
                st.session_state["checkerboard_result"] = (detection, grid, direct)
            else:
                detection["square_columns"] = square_columns
                detection["square_rows"] = square_rows
                detection["image_signature"] = image_signature
                detection["roi"] = roi
                st.session_state["checkerboard_result"] = (detection, None, None)
        except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
            st.error(f"Detection failed: {exc}")
    stored = st.session_state.get("checkerboard_result")
    if stored:
        stored_settings = stored[0]
        if (
            stored_settings.get("image_signature") != image_signature
            or stored_settings.get("square_columns") != square_columns
            or stored_settings.get("square_rows") != square_rows
            or tuple(stored_settings.get("roi", ())) != roi
        ):
            stored = None
            st.info("ROI or grid dimensions changed. Run detection again to refresh the measurements.")
    automatic_result = None
    suspect = False
    if stored:
        detection, grid, direct = stored
        st.write(
            f"Detected **{detection['detected_count']} / {detection['expected_count']}** internal intersections "
            f"using `{detection.get('method') or 'no detector'}`."
        )
        if not detection["success"]:
            st.warning(detection["failure_reason"])
        else:
            rms = detection["reprojection_rms_px"]
            cell_scale = float(np.median(np.linalg.norm(np.diff(grid, axis=1), axis=2)))
            suspect = rms > max(2.0, cell_scale * 0.02)
            st.write(f"Homography reprojection RMS: **{rms:.3f} px** (maximum {detection['reprojection_max_px']:.3f} px)")
            if suspect:
                st.warning("High grid residual: inspect the overlay. Final measurements require explicit confirmation.")
            else:
                st.success("Intersection count and grid residual passed basic checks.")
            preview_measurements = measure_grid(grid, direct)
            tentative_overlay = annotate_image(image, grid, direct, preview_measurements, True, suspect)
            tentative_preview = cv2.resize(tentative_overlay, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA) if preview_scale < 1 else tentative_overlay
            st.image(tentative_preview, caption="Proposed grid for review; orange boundaries are extrapolated.")
            accept = st.checkbox(
                "I reviewed the detected grid and accept these measurements",
                value=False,
                key="accept_detection",
            )
            if accept:
                automatic_result = (grid, direct, measure_grid(grid, direct), detection)
            else:
                st.info("Confirm the automatic grid before using its measurements.")

    st.subheader("Manual fallback")
    with st.expander("Estimate the grid from four outer board corners"):
        st.caption("Enter TL, TR, BR, BL in original-image pixels. A bilinear quadrilateral grid is estimated; these are not detected intersections.")
        defaults = [
            (left, top),
            (min(image_width - 1, right - 1), top),
            (min(image_width - 1, right - 1), min(image_height - 1, bottom - 1)),
            (left, min(image_height - 1, bottom - 1)),
        ]
        corner_values = []
        for index, (label, default) in enumerate(zip(("Top-left", "Top-right", "Bottom-right", "Bottom-left"), defaults)):
            c1, c2 = st.columns(2)
            corner_values.append((
                c1.number_input(f"{label} x", min_value=0.0, max_value=float(image_width - 1), value=float(default[0]), key=f"manual_x_{index}"),
                c2.number_input(f"{label} y", min_value=0.0, max_value=float(image_height - 1), value=float(default[1]), key=f"manual_y_{index}"),
            ))
        manual_columns, manual_rows = st.columns(2)
        mcols = int(manual_columns.number_input("Manual square columns", min_value=1, max_value=100, value=square_columns))
        mrows = int(manual_rows.number_input("Manual square rows", min_value=1, max_value=100, value=square_rows))
        if st.button("Measure estimated grid"):
            st.session_state.pop("use_manual_result", None)
            try:
                manual_corners = [(x, y) for x, y in corner_values]
                manual_grid = manual_board_grid(manual_corners, mcols, mrows)
                manual_direct = np.zeros(manual_grid.shape[:2], dtype=bool)
                manual_measurements = measure_grid(manual_grid, manual_direct, manual=True)
                st.session_state["manual_result"] = (
                    manual_grid, manual_direct, manual_measurements, mcols, mrows, image_signature
                )
            except (ValueError, cv2.error) as exc:
                st.error(f"Manual estimate failed: {exc}")

    result = automatic_result
    if "manual_result" in st.session_state:
        manual_grid, manual_direct, manual_measurements, result_mcols, result_mrows, manual_signature = st.session_state["manual_result"]
        if (
            manual_signature == image_signature
            and result_mcols == mcols
            and result_mrows == mrows
            and st.checkbox("Use manual perspective estimate (estimated, not detected)", key="use_manual_result")
        ):
            result = (
                manual_grid,
                manual_direct,
                manual_measurements,
                {
                    "method": "manual four-corner perspective model",
                    "square_columns": result_mcols,
                    "square_rows": result_mrows,
                    "points": [],
                    "reprojection_rms_px": None,
                    "roi": roi,
                },
            )

    if result is not None:
        grid, direct, measurements, result_metadata = result
        st.subheader("4. Inspect measurements")
        include_estimates = st.checkbox("Include estimated squares in summary statistics", value=True)
        summary = _summary(measurements, include_estimates)
        if summary.empty:
            st.warning("No squares are available for the selected summary filter.")
        else:
            st.metric("Squares measured", len(measurements))
            direct_count = int((measurements["measurement_source"] == "directly detected").sum())
            estimated_count = len(measurements) - direct_count
            st.caption(f"{direct_count} directly measured; {estimated_count} estimated/extrapolated.")
            st.dataframe(summary.round(3), use_container_width=True)
        show_labels = st.checkbox("Show square labels on overlay", value=True)
        is_manual = result_metadata.get("method") == "manual four-corner perspective model"
        overlay = annotate_image(image, grid, direct, measurements, show_labels, suspect and not is_manual)
        overlay_preview = cv2.resize(overlay, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA) if preview_scale < 1 else overlay
        st.image(overlay_preview, caption="Green: detected grid; orange: extrapolated boundary; red: directly detected intersections; yellow: suspect fit.")
        st.dataframe(measurements.round(3), use_container_width=True, height=450)
        if not is_manual and result_metadata.get("success"):
            with st.expander("Planar-homography diagnostic", expanded=False):
                try:
                    _show_planar_diagnostic(st, image, result_metadata, grid, direct)
                except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
                    st.error(f"Planar-homography diagnostic failed: {exc}")
        st.download_button("Download measurements CSV", csv_bytes(measurements), "checkerboard_measurements.csv", "text/csv")
        st.download_button("Download full-resolution annotated PNG", _png_bytes(overlay), "checkerboard_annotated.png", "image/png")
        metadata = {
            "detection_method": result_metadata.get("method"),
            "square_columns": result_metadata.get("square_columns"),
            "square_rows": result_metadata.get("square_rows"),
            "expected_intersections": result_metadata.get(
                "expected_count",
                (result_metadata.get("square_columns", 0) - 1)
                * (result_metadata.get("square_rows", 0) - 1),
            ),
            "detected_intersections": result_metadata.get("detected_count", 0),
            "detected_intersection_coordinates_px": result_metadata.get("points", []),
            "reprojection_rms_px": result_metadata.get("reprojection_rms_px"),
            "reprojection_max_px": result_metadata.get("reprojection_max_px"),
            "homography": result_metadata.get("homography"),
            "roi_original_px": result_metadata.get("roi"),
            "measurement_sources": measurements["measurement_source"].value_counts().to_dict(),
        }
        st.download_button(
            "Download measurement metadata JSON",
            json_bytes(image.shape, measurements, metadata),
            "checkerboard_measurements.json",
            "application/json",
        )
        st.caption("Legend: red dots are directly detected intersections; green grid segments are supported by detected points; orange segments/cells use extrapolated boundaries; yellow indicates a suspect fit.")


if __name__ == "__main__":
    main()
