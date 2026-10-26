import json
from io import BytesIO
from unittest.mock import patch
from urllib.error import URLError

import pytest

from ollama_summary import build_summary_context, summarize_with_ollama


def _analysis():
    import pandas as pd

    return {
        "metrics": {
            "corner_count": 4,
            "coverage_area_px2": 900.0,
            "coverage_fraction": 0.25,
            "rmse_px": 0.5,
            "median_px": 0.4,
            "p95_px": 0.8,
            "maximum_px": 0.9,
        },
        "records": pd.DataFrame(
            {
                "corner_id": ["R1C1", "R1C2", "R2C1", "R2C2"],
                "grid_row": [1, 1, 2, 2],
                "grid_column": [1, 2, 1, 2],
                "dx_px": [0.1, -0.2, 0.3, -0.4],
                "dy_px": [0.2, 0.1, -0.1, -0.2],
                "error_magnitude_px": [0.22, 0.22, 0.32, 0.45],
                "observed_x_px": [100, 120, 100, 120],
                "observed_y_px": [80, 80, 100, 100],
            }
        ),
    }


def test_summary_context_contains_aggregates_without_image_data():
    context = build_summary_context(_analysis())

    assert context["diagnostic"] == "deviation from planar homography"
    assert len(context["row_summaries"]) == 2
    assert len(context["column_summaries"]) == 2
    assert context["largest_corner_residuals"][0]["corner_id"] == "R2C2"
    assert "image" not in context


def test_ollama_request_uses_local_model_and_returns_text():
    response_bytes = json.dumps(
        {"message": {"content": "The measured deviation is small."}}
    ).encode("utf-8")
    with patch("ollama_summary.urlopen", return_value=BytesIO(response_bytes)) as urlopen:
        summary = summarize_with_ollama(_analysis(), " qwen3.5:4b ")

    assert summary == "The measured deviation is small."
    request = urlopen.call_args.args[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "http://localhost:11434/api/chat"
    assert payload["model"] == "qwen3.5:4b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"]["num_predict"] == 256
    assert "Do not recalculate values" in payload["messages"][0]["content"]
    assert "observed_x_px" not in payload["messages"][1]["content"]


def test_ollama_connection_failure_is_reported_clearly():
    with patch("ollama_summary.urlopen", side_effect=URLError("offline")):
        with pytest.raises(RuntimeError, match="Ensure Ollama is running"):
            summarize_with_ollama(_analysis(), "qwen3.5:4b")


@pytest.mark.parametrize("model", ["", "  "])
def test_missing_model_name_is_rejected(model):
    with pytest.raises(ValueError, match="model name"):
        summarize_with_ollama(_analysis(), model)


def test_empty_model_response_is_not_presented_as_success():
    response_bytes = json.dumps({"message": {"content": "  "}}).encode("utf-8")
    with patch("ollama_summary.urlopen", return_value=BytesIO(response_bytes)):
        with pytest.raises(RuntimeError, match="empty summary"):
            summarize_with_ollama(_analysis(), "qwen3.5:4b")
