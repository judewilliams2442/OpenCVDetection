# Checkerboard Pixel Measurement Tool

A local Streamlit application for measuring checkerboard squares in image-frame
pixels. Processing is performed on the computer running the application; images
are not sent to a cloud service.

## Setup

Use Python 3.10 or newer. From this directory, install the dependencies and
launch the interface:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Using `python -m streamlit` runs Streamlit from the same Python environment
where the requirements were installed, even when the `streamlit` command is
not on the Windows `PATH`. If Python reports that it cannot find the module,
install into that interpreter with `python -m pip install -r requirements.txt`
and retry.

To run the included tests:

```powershell
python -m pip install pytest
python -m pytest
```

## Workflow

1. Upload a PNG, JPEG, or TIFF frame. The displayed dimensions are the original
   image dimensions; all output coordinates and distances use those pixels.
2. Set the crop using the horizontal **left–right** and vertical **top–bottom**
   range sliders. Drag either slider handle to adjust a bound, or click its
   numeric value to enter an exact original-frame pixel. The blue outline shows
   the selected region; the initial bounds include the full image. Right and
   bottom bounds are exclusive.
3. Enter **square columns** and **square rows** (the total counts of full squares,
   not intersections). A board with `C` columns and `R` rows has
   `(C - 1) × (R - 1)` internal intersections. Use “Help estimate grid
   dimensions” to generate candidates, then confirm a candidate against the
   visible board and enter its dimensions.
4. Choose **Detect checkerboard**. Review the detected intersection count,
   reprojection residual, and annotated overlay. Accept automatic measurements
   only after inspecting the overlay. Green indicates grid segments supported
   by direct detections; orange indicates extrapolated boundary segments; red
   markers show directly detected intersections; yellow marks a suspect fit.
5. If detection does not work, expand the manual fallback and enter the outer
   board corners in top-left, top-right, bottom-right, bottom-left order. This
   bilinearly interpolated quadrilateral grid is labeled an estimate, not a
   direct detection.
6. Inspect the per-square table and summary, optionally exclude estimates from
   summary statistics, and download CSV, full-resolution PNG, and JSON exports.
   The label toggle controls whether square IDs appear on the overlay.
7. Expand **Planar-homography diagnostic** after accepting an automatic
   detection to compare original detected intersections with a best-fit ideal
   unit grid under perspective. This reports **deviation from planar
   homography**, not lens distortion; generated boundary points remain visible
   separately and are not included in the fit or residual statistics.

Each measured square reports its four corners, individual edge lengths, mean
horizontal and vertical dimensions, diagonals, area, and measurement source.
Exported values retain floating-point precision; the displayed tables are
rounded for readability.

## Limitations

- A planar homography models perspective but not lens distortion or a warped,
  non-planar board. High residuals should be investigated rather than accepted
  automatically.
- Internal chessboard intersections are directly detected; the outer board
  boundary is extrapolated. Consequently, outermost squares are estimates and
  are explicitly identified as such. Cropped, occluded, irregular, or strongly
  distorted boards may not support reliable extrapolation.
- Automatically detected internal intersections are measured at their
  subpixel image locations; they are not forced into ideal square or parallel
  shapes. Perspective and a non-flat board can make projected cells skew or
  vary across the image. Review individual corner markers against the visible
  black/white junctions, particularly where wrinkles or seams cross the board.
- The planar-homography diagnostic fits only unique, genuinely detected
  internal corners in original-frame pixels. Its sparse residual heatmap,
  corner table, row/column summaries, and CSV describe departures from a
  planar perspective grid; optional RANSAC classification is an additional
  comparison and does not remove points from baseline metrics.
- The manual fallback uses bilinear interpolation from only the four supplied
  outer corners. It estimates, rather than detects, intermediate intersections
  and is less geometrically constrained than the automatic homography fit.
- Pixel dimensions vary across an image with perspective, viewing angle, and
  lens distortion even when the physical squares are identical. This tool does
  not convert pixels to physical units.
- The app limits inputs to 100 million pixels and reduces only the detection
  working image when needed. Measurements and downloadable annotation remain
  in original-frame coordinates and resolution.
