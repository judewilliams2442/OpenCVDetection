"""PDF report generation for checkerboard measurement results."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from textwrap import fill
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from planar_diagnostic import analyze_axis_nonlinearity, analyze_planar_deviation


_SUMMARY_METRICS = (
    ("average_width_px", "Mean horizontal cell dimension (px)"),
    ("average_height_px", "Mean vertical cell dimension (px)"),
    ("area_px2", "Cell area (px2)"),
    ("top_edge_px", "Top edge length (px)"),
    ("bottom_edge_px", "Bottom edge length (px)"),
    ("left_edge_px", "Left edge length (px)"),
    ("right_edge_px", "Right edge length (px)"),
    ("diagonal_tl_br_px", "TL-BR diagonal (px)"),
    ("diagonal_tr_bl_px", "TR-BL diagonal (px)"),
)

_APPENDIX_COLUMNS = (
    "square_id",
    "top_left_x_px",
    "top_left_y_px",
    "average_width_px",
    "average_height_px",
    "top_edge_px",
    "bottom_edge_px",
    "left_edge_px",
    "right_edge_px",
    "area_px2",
    "measurement_source",
)

_NAVY = "#17324D"
_BLUE = "#326B8C"
_PALE_BLUE = "#EAF2F7"
_PALE_GRAY = "#F4F6F8"
_TEXT = "#243447"
_MUTED = "#5B6875"


def _summary_table(measurements: pd.DataFrame) -> pd.DataFrame:
    if measurements.empty:
        return pd.DataFrame()
    rows = []
    for column, label in _SUMMARY_METRICS:
        values = measurements[column].to_numpy(dtype=np.float64)
        rows.append(
            {
                "Measure": label,
                "N": len(values),
                "Mean": float(np.mean(values)),
                "SD": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "Median": float(np.median(values)),
                "Min": float(np.min(values)),
                "Max": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def _format_coefficients(coefficients: np.ndarray, variable: str) -> str:
    terms = []
    for degree, coefficient in enumerate(coefficients):
        power = "" if degree == 0 else f" * {variable}^{degree}"
        terms.append(f"{coefficient:+.4g}{power}")
    return " ".join(terms).lstrip("+")


def _add_footer(figure: Any, page_number: int) -> None:
    figure.text(
        0.05,
        0.022,
        "CHECKERBOARD IMAGE ANALYSIS",
        ha="left",
        fontsize=7,
        color=_MUTED,
        weight="bold",
    )
    figure.text(
        0.95,
        0.022,
        f"{page_number}",
        ha="right",
        fontsize=8,
        color=_MUTED,
    )


def _add_page_header(figure: Any, title: str, subtitle: str | None = None) -> None:
    figure.add_artist(
        Rectangle(
            (0, 0.89),
            1,
            0.11,
            transform=figure.transFigure,
            facecolor=_NAVY,
            edgecolor="none",
            zorder=-1,
        )
    )
    figure.text(
        0.055,
        0.948,
        title,
        ha="left",
        va="center",
        color="white",
        fontsize=17,
        weight="bold",
    )
    if subtitle:
        figure.text(
            0.057,
            0.908,
            subtitle,
            ha="left",
            va="center",
            color="#D9E6EF",
            fontsize=8,
        )


def _add_card(
    figure: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    value: str,
) -> None:
    figure.add_artist(
        Rectangle(
            (x, y),
            width,
            height,
            transform=figure.transFigure,
            facecolor=_PALE_BLUE,
            edgecolor="#D3E0E8",
            linewidth=0.7,
            zorder=-1,
        )
    )
    figure.text(
        x + 0.014,
        y + height * 0.68,
        label.upper(),
        ha="left",
        va="center",
        fontsize=6.5,
        weight="bold",
        color=_MUTED,
    )
    figure.text(
        x + 0.014,
        y + height * 0.30,
        value,
        ha="left",
        va="center",
        fontsize=13,
        weight="bold",
        color=_NAVY,
    )


def build_analysis_report_pdf(
    image_rgb: np.ndarray,
    annotated_rgb: np.ndarray,
    measurements: pd.DataFrame,
    detection: dict[str, Any],
    roi: tuple[int, int, int, int],
    image_name: str,
    include_estimates: bool = True,
) -> bytes:
    """Create a multi-page summary, spatial analysis, trend review, and data appendix."""
    if measurements.empty:
        raise ValueError("Cannot generate a report without square measurements.")
    selected = measurements
    if not include_estimates:
        selected = measurements[
            measurements["measurement_source"] == "directly detected"
        ]

    direct_count = int(
        measurements["measurement_source"].eq("directly detected").sum()
    )
    estimated_count = len(measurements) - direct_count
    is_manual = detection.get("method") == "manual four-corner perspective model"
    planar_analysis = None
    axis_analysis: dict[str, dict[str, Any]] = {}
    if detection.get("success") and not is_manual:
        planar_analysis = analyze_planar_deviation(detection)
        axis_analysis = analyze_axis_nonlinearity(planar_analysis)

    buffer = BytesIO()
    summary = _summary_table(selected)
    image_height, image_width = image_rgb.shape[:2]
    figure_number = 0

    pattern = detection.get("pattern_size", (0, 0))
    columns = int(detection.get("square_columns", pattern[0] + 1))
    rows = int(detection.get("square_rows", pattern[1] + 1))
    requested_columns = detection.get("requested_square_columns", columns)
    requested_rows = detection.get("requested_square_rows", rows)
    method = detection.get("method", "not recorded")
    roi_x, roi_y, roi_width, roi_height = roi

    with PdfPages(buffer) as pdf:
        figure_number += 1
        figure = plt.figure(figsize=(11.7, 8.3), facecolor="white")
        _add_page_header(
            figure,
            "Checkerboard Image Analysis",
            f"{image_name}   |   Generated {datetime.now().astimezone():%Y-%m-%d %H:%M %Z}",
        )
        for bounds, data, title in (
            ([0.055, 0.535, 0.425, 0.29], image_rgb, "SOURCE IMAGE"),
            ([0.52, 0.535, 0.425, 0.29], annotated_rgb, "DETECTED MEASUREMENT GRID"),
        ):
            axis = figure.add_axes(bounds)
            axis.imshow(data)
            axis.set_title(title, loc="left", fontsize=8, color=_MUTED, pad=7, weight="bold")
            axis.axis("off")

        mean_width = float(measurements["average_width_px"].mean())
        mean_height = float(measurements["average_height_px"].mean())
        for index, (label, value) in enumerate(
            (
                ("MEASURED CELLS", f"{len(measurements):,}"),
                ("DETECTED CORNERS", f"{detection.get('detected_count', 0):,}"),
                ("MEAN WIDTH", f"{mean_width:.2f} px"),
                ("MEAN HEIGHT", f"{mean_height:.2f} px"),
            )
        ):
            _add_card(figure, 0.055 + index * 0.226, 0.425, 0.211, 0.078, label, value)

        source_counts = measurements["measurement_source"].value_counts()
        source_text = ", ".join(f"{key}: {value}" for key, value in source_counts.items())
        if detection.get("visible_only"):
            coverage_text = "Partial edge cells excluded; only cells with four detected corners are measured."
        else:
            coverage_text = "Estimated boundary cells are marked separately from direct measurements."
        if planar_analysis is not None:
            metrics = planar_analysis["metrics"]
            residual_text = (
                f"Planar residual RMSE {metrics['rmse_px']:.3f} px; "
                f"median {metrics['median_px']:.3f}; p95 {metrics['p95_px']:.3f}; "
                f"max {metrics['maximum_px']:.3f}; frame coverage "
                f"{metrics['coverage_fraction']:.1%}."
            )
        else:
            residual_text = "Automatic corner residual analysis unavailable for a manual grid estimate."
        details = [
            f"Image: {image_width} x {image_height} px",
            f"Region: x={roi_x}, y={roi_y}, width={roi_width}, height={roi_height} px",
            f"Requested grid: {requested_columns} x {requested_rows} squares",
            f"Measured grid: {columns} x {rows} squares",
            f"Detector: {method}",
            f"Sources: {source_text}",
            coverage_text,
            residual_text,
            f"Summary includes estimated cells: {'yes' if include_estimates else 'no'}. Appendix includes every measured cell.",
            "Pixel measurements only; no physical scale or camera calibration supplied.",
        ]
        detail_axis = figure.add_axes([0.055, 0.075, 0.425, 0.32])
        detail_axis.set_facecolor(_PALE_GRAY)
        detail_axis.set_xticks([])
        detail_axis.set_yticks([])
        for spine in detail_axis.spines.values():
            spine.set_color("#E1E6EB")
        detail_axis.text(
            0.04,
            0.94,
            "ANALYSIS SCOPE",
            transform=detail_axis.transAxes,
            fontsize=8,
            color=_BLUE,
            weight="bold",
            va="top",
        )
        wrapped_details = "\n".join(f"• {fill(item, width=58)}" for item in details)
        detail_axis.text(
            0.04,
            0.84,
            wrapped_details,
            transform=detail_axis.transAxes,
            fontsize=7.1,
            color=_TEXT,
            linespacing=1.25,
            va="top",
        )

        table_axis = figure.add_axes([0.52, 0.075, 0.425, 0.32])
        table_axis.axis("off")
        table_axis.text(
            0,
            1.02,
            "MEASUREMENT SUMMARY",
            transform=table_axis.transAxes,
            fontsize=8,
            color=_BLUE,
            weight="bold",
            va="bottom",
        )
        if summary.empty:
            table_axis.text(
                0,
                0.9,
                "No directly detected samples remain in the selected summary.",
                transform=table_axis.transAxes,
                fontsize=8,
                va="top",
            )
        else:
            table = table_axis.table(
                cellText=[
                    [
                        row["Measure"],
                        row["N"],
                        f"{row['Mean']:.2f}",
                        f"{row['SD']:.2f}",
                        f"{row['Median']:.2f}",
                        f"{row['Min']:.2f}",
                        f"{row['Max']:.2f}",
                    ]
                    for _, row in summary.iterrows()
                ],
                colLabels=["Measurement", "N", "Mean", "SD", "Median", "Min", "Max"],
                colWidths=[0.36, 0.07, 0.11, 0.10, 0.13, 0.11, 0.12],
                bbox=[0, 0, 1, 0.94],
                cellLoc="right",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(5.8)
            for (row_index, _), cell in table.get_celld().items():
                cell.set_edgecolor("#D5DDE4")
                cell.set_linewidth(0.35)
                if row_index == 0:
                    cell.set_facecolor(_NAVY)
                    cell.get_text().set_color("white")
                    cell.get_text().set_weight("bold")
                elif row_index % 2 == 0:
                    cell.set_facecolor(_PALE_GRAY)
        _add_footer(figure, figure_number)
        pdf.savefig(figure)
        plt.close(figure)

        figure_number += 1
        figure = plt.figure(figsize=(11.7, 8.3), facecolor="white")
        _add_page_header(
            figure,
            "Spatial Measurement Maps",
            "Each square is colored by its measured projected dimension in image pixels.",
        )
        for x, column, title, color_map, colorbar_x in (
            (0.075, "average_width_px", "Horizontal cell dimension", "viridis", 0.445),
            (0.555, "average_height_px", "Vertical cell dimension", "magma", 0.925),
        ):
            matrix = measurements.pivot(index="row", columns="column", values=column)
            axis = figure.add_axes([x, 0.20, 0.34, 0.62])
            image = axis.imshow(
                matrix.to_numpy(),
                origin="upper",
                aspect="auto",
                cmap=color_map,
                interpolation="nearest",
            )
            axis.set_title(title, fontsize=10, color=_NAVY, pad=10, weight="bold")
            axis.set_xlabel("Square column", fontsize=8, labelpad=5)
            axis.set_ylabel("Square row", fontsize=8, labelpad=5)
            axis.set_xticks(np.arange(0, len(matrix.columns), 2))
            axis.set_xticklabels(matrix.columns[::2], fontsize=6.5)
            axis.set_yticks(np.arange(len(matrix.index)))
            axis.set_yticklabels(matrix.index, fontsize=6.5)
            axis.tick_params(length=0)
            colorbar_axis = figure.add_axes([colorbar_x, 0.28, 0.012, 0.46])
            figure.colorbar(image, cax=colorbar_axis).ax.tick_params(labelsize=6.5)
            colorbar_axis.set_ylabel("Pixels", fontsize=7, labelpad=5)
        figure.add_artist(
            Rectangle(
                (0.075, 0.075),
                0.85,
                0.07,
                transform=figure.transFigure,
                facecolor=_PALE_GRAY,
                edgecolor="none",
            )
        )
        figure.text(
            0.09,
            0.11,
            "Maps show projected image dimensions, not physical square size. Variation can reflect "
            "perspective, lens distortion, board curvature, or measurement uncertainty.",
            fontsize=7.5,
            color=_TEXT,
            va="center",
        )
        _add_footer(figure, figure_number)
        pdf.savefig(figure)
        plt.close(figure)

        figure_number += 1
        figure = plt.figure(figsize=(11.7, 8.3), facecolor="white")
        _add_page_header(
            figure,
            "X/Y Residual Trend Diagnostics",
            "Observed corner locations minus their best-fit planar homography prediction.",
        )
        if planar_analysis is None:
            figure.text(
                0.08,
                0.68,
                fill(
                    "This analysis is unavailable for a manual four-corner estimate because it "
                    "contains no independently detected intersections. A manual grid cannot "
                    "establish whether residuals follow a nonlinear trend or appear random.",
                    width=105,
                ),
                fontsize=12,
                color=_TEXT,
                va="top",
            )
        else:
            for x, name, title in (
                (0.075, "x", "Horizontal residual (dx) vs image X"),
                (0.555, "y", "Vertical residual (dy) vs image Y"),
            ):
                result = axis_analysis[name]
                axis = figure.add_axes([x, 0.20, 0.37, 0.61])
                axis.set_title(title, fontsize=9.5, color=_NAVY, pad=10, weight="bold")
                axis.set_xlabel(f"Normalized image {name.upper()} position", fontsize=8)
                axis.set_ylabel(f"{'dx' if name == 'x' else 'dy'} residual (px)", fontsize=8)
                axis.grid(True, color="#D9E0E6", linewidth=0.55, alpha=0.8)
                axis.axhline(0, color="#697586", linewidth=0.8)
                if not result["available"]:
                    axis.text(
                        0.5,
                        0.5,
                        fill(result["interpretation"], width=42),
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                        fontsize=8,
                        color=_TEXT,
                    )
                    continue
                coordinate = result["normalized_coordinates"]
                residual = result["residuals_px"]
                coefficients = result["coefficients_px"]
                axis.scatter(
                    coordinate,
                    residual,
                    s=13,
                    alpha=0.62,
                    color=_BLUE,
                    edgecolors="white",
                    linewidths=0.25,
                    zorder=2,
                )
                curve_x = np.linspace(float(coordinate.min()), float(coordinate.max()), 200)
                axis.plot(
                    curve_x,
                    np.polynomial.polynomial.polyval(curve_x, coefficients),
                    color="#D45D3F",
                    linewidth=2.2,
                    label=f"degree {result['best_degree']} fit",
                    zorder=3,
                )
                axis.legend(loc="lower right", fontsize=7, frameon=False)
                axis.text(
                    0.03,
                    0.97,
                    f"{result['sample_count']} corners | blocked CV",
                    transform=axis.transAxes,
                    va="top",
                    fontsize=7,
                    color=_MUTED,
                )
        figure.add_artist(
            Rectangle(
                (0.075, 0.075),
                0.85,
                0.07,
                transform=figure.transFigure,
                facecolor=_PALE_GRAY,
                edgecolor="none",
            )
        )
        figure.text(
            0.09,
            0.11,
            "Points are detected corners. The colored curve is a low-order fit; model selection and "
            "interpretation are summarized on the next page.",
            fontsize=7.5,
            color=_TEXT,
            va="center",
        )
        _add_footer(figure, figure_number)
        pdf.savefig(figure)
        plt.close(figure)

        figure_number += 1
        figure = plt.figure(figsize=(11.7, 8.3), facecolor="white")
        _add_page_header(
            figure,
            "Trend Findings & Interpretation",
            "Spatially blocked cross-validation compares constant, linear, quadratic, and cubic fits.",
        )
        for x, name, label in ((0.055, "x", "X-AXIS RESIDUAL"), (0.52, "y", "Y-AXIS RESIDUAL")):
            axis = figure.add_axes([x, 0.48, 0.425, 0.32])
            axis.set_facecolor(_PALE_GRAY)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#DDE4EA")
            result = axis_analysis.get(name, {"available": False, "interpretation": "Not available."})
            axis.text(
                0.04,
                0.92,
                label,
                transform=axis.transAxes,
                va="top",
                fontsize=8,
                color=_BLUE,
                weight="bold",
            )
            if result["available"]:
                cv_errors = result["cross_validated_rmse_px"]
                cv_text = "    ".join(
                    f"degree {degree}: {error:.3f} px"
                    for degree, error in cv_errors.items()
                )
                equation = (
                    f"Approximation: {_format_coefficients(result['coefficients_px'], 'u')}"
                )
                body = "\n".join(
                    (
                        fill(result["interpretation"], width=52),
                        "",
                        fill(f"Blocked CV RMSE: {cv_text}", width=52),
                        "",
                        fill(equation, width=52),
                        "",
                        f"Best degree: {result['best_degree']} | "
                        f"gain vs linear: {result['improvement_over_linear']:.1%}",
                    )
                )
            else:
                body = fill(result["interpretation"], width=52)
            axis.text(
                0.04,
                0.78,
                body,
                transform=axis.transAxes,
                va="top",
                fontsize=7.3,
                linespacing=1.35,
                color=_TEXT,
            )

        figure.add_artist(
            Rectangle(
                (0.055, 0.17),
                0.89,
                0.24,
                transform=figure.transFigure,
                facecolor="#FFF7E8",
                edgecolor="#E9D6AE",
                linewidth=0.7,
            )
        )
        caveat = (
            "WHAT THIS CAN AND CANNOT SAY\n"
            "A polynomial that predicts held-out grid bands better than simpler fits is evidence "
            "of a repeatable spatial residual pattern. It is not, by itself, proof of lens distortion. "
            "One image cannot separate lens behavior from board curvature or wrinkles, perspective-model "
            "mismatch, uneven lighting, or corner-localization error. Likewise, failure to find a smooth "
            "trend does not prove the remaining variation is random. Repeat captures with a rigid flat "
            "board at multiple positions and rotations are needed to isolate camera effects."
        )
        figure.text(
            0.075,
            0.375,
            fill(caveat, width=135),
            fontsize=8,
            color=_TEXT,
            va="top",
            linespacing=1.4,
        )
        if planar_analysis is not None:
            metrics = planar_analysis["metrics"]
            figure.text(
                0.06,
                0.11,
                f"Planar fit: RMSE {metrics['rmse_px']:.3f} px | median {metrics['median_px']:.3f} px | "
                f"95th percentile {metrics['p95_px']:.3f} px | maximum {metrics['maximum_px']:.3f} px | "
                f"coverage {metrics['coverage_fraction']:.1%}",
                fontsize=8,
                color=_NAVY,
                weight="bold",
            )
        _add_footer(figure, figure_number)
        pdf.savefig(figure)
        plt.close(figure)

        appendix = measurements.loc[:, _APPENDIX_COLUMNS].copy().round(3)
        appendix["measurement_source"] = appendix["measurement_source"].replace(
            {
                "directly detected": "direct",
                "extrapolated boundary estimate": "estimated",
                "manual perspective estimate": "manual",
            }
        )
        rows_per_page = 24
        widths = [0.075, 0.09, 0.09, 0.095, 0.095, 0.085, 0.085, 0.08, 0.08, 0.105, 0.10]
        for start in range(0, len(appendix), rows_per_page):
            figure_number += 1
            chunk = appendix.iloc[start : start + rows_per_page]
            figure = plt.figure(figsize=(11.7, 8.3), facecolor="white")
            _add_page_header(
                figure,
                "Per-square Measurement Appendix",
                f"Records {start + 1:,}–{start + len(chunk):,} of {len(appendix):,} | "
                "Dimensions in pixels; area in square pixels.",
            )
            axis = figure.add_axes([0.045, 0.09, 0.91, 0.75])
            axis.axis("off")
            table = axis.table(
                cellText=chunk.to_numpy().tolist(),
                colLabels=[
                    "Square",
                    "TL X",
                    "TL Y",
                    "Mean W",
                    "Mean H",
                    "Top",
                    "Bottom",
                    "Left",
                    "Right",
                    "Area",
                    "Source",
                ],
                colWidths=widths,
                bbox=[0, 0, 1, 1],
                cellLoc="right",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            for (row_index, column_index), cell in table.get_celld().items():
                cell.set_edgecolor("#D8E0E6")
                cell.set_linewidth(0.35 if row_index else 0.6)
                if row_index == 0:
                    cell.set_facecolor(_NAVY)
                    cell.get_text().set_color("white")
                    cell.get_text().set_weight("bold")
                    cell.get_text().set_ha("center")
                else:
                    cell.set_facecolor("white" if row_index % 2 else _PALE_GRAY)
                    if column_index in (0, 10):
                        cell.get_text().set_ha("center")
            _add_footer(figure, figure_number)
            pdf.savefig(figure)
            plt.close(figure)

    return buffer.getvalue()
