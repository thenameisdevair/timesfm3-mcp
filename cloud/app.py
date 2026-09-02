import os
import sys
from pathlib import Path

import torch
from fastmcp import FastMCP
from huggingface_hub import login
from timesfm3 import ModelConfig, TimesFM3Evaluator

ROOT = Path(__file__).resolve().parents[1]
for path in (Path(__file__).resolve().parent, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forecasting import run_forecast  # noqa: E402

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    sys.stderr.write("Authenticating with Hugging Face...\n")
    login(token=hf_token)
else:
    sys.stderr.write("WARNING: HF_TOKEN is not set; gated weight download may fail.\n")

mcp = FastMCP("TimesFM-3")

sys.stderr.write("Loading TimesFM-3 into memory...\n")
device = "cuda" if torch.cuda.is_available() else "cpu"
sys.stderr.write(f"Device: {device}\n")
config = ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", device=device)
forecaster = TimesFM3Evaluator(config)
sys.stderr.write("Model ready.\n")


@mcp.tool()
def forecast(
    series: list[list[float]] | None = None,
    history: list[float] | None = None,
    horizon: int = 5,
    series_ids: list[str] | None = None,
    past_covariates: list[list[float]] | None = None,
    future_covariates: list[list[float]] | None = None,
) -> dict:
    """Zero-shot forecast with Google TimesFM-3.

    Pass one series (univariate) or several related series (joint multivariate).
    Optional past_covariates must be length T. Optional future_covariates
    (known ahead, e.g. promo flags) must be length T + horizon.

    Returns a median point forecast plus nine quantile bands (q10 through q90)
    per series.

    TimesFM-3 weights are licensed for non-commercial, non-production use only.

    Args:
        series: Target series, each a chronological list of the same length T.
            One row is univariate. Two or more rows are forecast jointly.
        history: Back-compat shortcut for a single series. Do not pass with series.
        horizon: Number of future steps. Must be >= 1.
        series_ids: Optional names, one per series row.
        past_covariates: Channels known only in the past. Each length T.
        future_covariates: Channels known in the past and future. Each length T+horizon.
    """
    return run_forecast(
        forecaster,
        series=series,
        history=history,
        horizon=horizon,
        series_ids=series_ids,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
    )


@mcp.tool()
def forecast_demand(history: list[float], horizon: int = 5) -> dict:
    """Deprecated alias of `forecast` for a single series. Prefer `forecast`.

    Same TimesFM-3 non-commercial license limit.
    """
    return run_forecast(forecaster, history=history, horizon=horizon)


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=7860)
