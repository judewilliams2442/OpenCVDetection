"""Local Ollama summaries of computed planar-homography diagnostics."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


def build_summary_context(analysis: dict[str, Any]) -> dict[str, Any]:
    """Prepare compact computed results for interpretation by a local model."""
    records = analysis["records"]
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
    top_errors = records.nlargest(8, "error_magnitude_px")[
        ["corner_id", "grid_row", "grid_column", "dx_px", "dy_px", "error_magnitude_px"]
    ]
    return {
        "diagnostic": "deviation from planar homography",
        "metrics": analysis["metrics"],
        "row_summaries": row_summary.round(4).to_dict(orient="records"),
        "column_summaries": column_summary.round(4).to_dict(orient="records"),
        "largest_corner_residuals": top_errors.round(4).to_dict(orient="records"),
        "ransac_classification": (
            records["ransac_classification"].value_counts().to_dict()
            if "ransac_classification" in records
            else None
        ),
    }


def summarize_with_ollama(
    analysis: dict[str, Any],
    model: str,
    timeout_seconds: float = 120,
) -> str:
    """Ask an Ollama model on localhost to explain computed results."""
    model_name = model.strip()
    if not model_name:
        raise ValueError("Enter an installed Ollama model name.")
    if timeout_seconds <= 0:
        raise ValueError("The Ollama request timeout must be positive.")
    context = json.dumps(build_summary_context(analysis), allow_nan=False)
    payload = {
        "model": model_name,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You explain checkerboard measurements to a non-expert using only the "
                    "provided computed results. The residual is observed pixel position minus "
                    "the fitted planar-homography prediction; no image correction is applied. "
                    "Do not recalculate values, invent patterns, or diagnose physical causes. "
                    "Do not label the board, camera, or measurements accurate or inaccurate. "
                    "Only compare rows or columns when the supplied summaries clearly support "
                    "the comparison. Do not extrapolate beyond the measured corners or coverage. "
                    "Describe the result as 'deviation from planar homography', never as lens "
                    "distortion. State limitations and uncertainty, and mention that a planar "
                    "fit is not a lens-calibration or image-correction procedure. If results "
                    "do not support a conclusion, say so."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize these computed diagnostic results in 3-6 concise bullets, "
                    "under 180 words:\n"
                    f"{context}"
                ),
            },
        ],
        "think": False,
        "options": {"num_predict": 256},
    }
    request = Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Ollama returned HTTP {exc.code}"
            + (f": {detail}" if detail else ".")
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            "Could not connect to Ollama at localhost:11434. Ensure Ollama is running."
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Ollama did not respond within {timeout_seconds:g} seconds."
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Ollama returned an invalid JSON response.") from exc

    message = result.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Ollama returned an empty summary.")
    return content.strip()
