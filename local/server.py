import sys

import numpy as np
import torch
from fastmcp import FastMCP
from timesfm3 import ModelConfig, TimesFM3Evaluator

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
LICENSE_NOTE = (
    "TimesFM-3 pretrained weights (google/timesfm-3.0-pytorch) are released "
    "under Google's timesfm-non-commercial-license-v1.0. They may be used for "
    "research, evaluation, and non-production experiments only — not for "
    "commercial or production systems. This MCP server's own code is Apache-2.0."
)

mcp = FastMCP("TimesFM-3")

sys.stderr.write("Loading TimesFM-3 into memory...\n")
device = "cuda" if torch.cuda.is_available() else "cpu"
sys.stderr.write(f"Device: {device}\n")
config = ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", device=device)
forecaster = TimesFM3Evaluator(config)
sys.stderr.write("Model ready.\n")


def _serialize_quantiles(quantiles) -> dict | None:
    if quantiles is None:
        return None
    q = np.asarray(quantiles)
    if q.ndim == 1:
        q = q.reshape(-1, 1)
    # Official univariate shape is (horizon, 9).
    packed = {}
    n_q = int(q.shape[-1])
    for i in range(n_q):
        if i < len(QUANTILE_LEVELS):
            key = f"q{int(QUANTILE_LEVELS[i] * 100)}"
        else:
            key = f"q_index_{i}"
        packed[key] = q[:, i].astype(float).tolist()
    return packed


def _run_forecast(history: list[float], horizon: int) -> dict:
    if not history:
        return {"status": "error", "error": "history must be a non-empty list of numbers"}
    if horizon < 1:
        return {"status": "error", "error": "horizon must be >= 1"}

    history_array = np.asarray(history, dtype=np.float32).reshape(-1)
    sys.stderr.write(f"Inference: {history_array.size} points -> {horizon} steps\n")

    outputs = list(
        forecaster.predict_batch(
            [history_array],
            horizon=horizon,
            return_quantiles=True,
            use_symmetric_averaging=False,
        )
    )
    first = outputs[0]
    return {
        "status": "success",
        "model": "google/timesfm-3.0-pytorch",
        "context_length": int(history_array.size),
        "horizon": int(horizon),
        "forecast": np.asarray(first.forecast).astype(float).tolist(),
        "quantiles": _serialize_quantiles(first.quantiles),
        "quantile_levels": QUANTILE_LEVELS,
        "license": LICENSE_NOTE,
    }


@mcp.tool()
def forecast(history: list[float], horizon: int = 5) -> dict:
    """Zero-shot forecast with Google TimesFM-3.

    Returns a median point forecast plus nine quantile bands (q10 through q90).

    TimesFM-3 weights are licensed for non-commercial, non-production use only.
    Do not call this tool as part of a paid product, customer deliverable, or
    production planning system.

    Args:
        history: Observed values in chronological order. Longer context is better.
        horizon: Number of future steps to predict. Must be >= 1.
    """
    return _run_forecast(history, horizon)


@mcp.tool()
def forecast_demand(history: list[float], horizon: int = 5) -> dict:
    """Deprecated alias of `forecast`. Prefer `forecast`.

    Same contract and the same TimesFM-3 non-commercial license limit.
    """
    return _run_forecast(history, horizon)


if __name__ == "__main__":
    mcp.run()
