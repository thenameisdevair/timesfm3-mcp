"""TimesFM-3 forecast helpers shared by the local and cloud MCP servers.

Shapes match Google's official TimesFM 3.0 API:

- Univariate context: 1D array of length T
- Multivariate context: 2D array of shape (V, T)
- past_only_covariates: (C_past, T)
- past_future_covariates: (C_future, T + H)
- Univariate output: forecast (H,), quantiles (H, 9)
- Multivariate output: forecast (V, H), quantiles (V, H, 9)
"""

from __future__ import annotations

from typing import Any

import numpy as np

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CHECKPOINT = "google/timesfm-3.0-pytorch"
LICENSE_NOTE = (
    "TimesFM-3 pretrained weights (google/timesfm-3.0-pytorch) are released "
    "under Google's timesfm-non-commercial-license-v1.0. They may be used for "
    "research, evaluation, and non-production experiments only — not for "
    "commercial or production systems. This MCP server's own code is Apache-2.0."
)


def serialize_quantiles(quantiles: Any) -> dict[str, list[float]] | None:
    if quantiles is None:
        return None
    q = np.asarray(quantiles)
    if q.ndim == 1:
        q = q.reshape(-1, 1)
    if q.ndim != 2:
        raise ValueError(f"quantiles must be 1D or 2D (H, Q), got shape {q.shape}")
    packed: dict[str, list[float]] = {}
    n_q = int(q.shape[-1])
    for i in range(n_q):
        if i < len(QUANTILE_LEVELS):
            key = f"q{int(round(QUANTILE_LEVELS[i] * 100))}"
        else:
            key = f"q_index_{i}"
        packed[key] = q[:, i].astype(float).tolist()
    return packed


def resolve_series(
    series: list[list[float]] | None,
    history: list[float] | None,
) -> np.ndarray:
    if series is not None and history is not None:
        raise ValueError("pass series or history, not both")
    if history is not None:
        if not history:
            raise ValueError("history must be a non-empty list of numbers")
        return np.asarray(history, dtype=np.float32).reshape(1, -1)
    if series is None or not series:
        raise ValueError("series must be a non-empty list of series")
    rows: list[np.ndarray] = []
    lengths: list[int] = []
    for i, row in enumerate(series):
        if not row:
            raise ValueError(f"series[{i}] must be a non-empty list of numbers")
        arr = np.asarray(row, dtype=np.float32).reshape(-1)
        rows.append(arr)
        lengths.append(int(arr.size))
    if len(set(lengths)) != 1:
        raise ValueError("all series rows must have the same length (context T)")
    return np.stack(rows, axis=0)


def _as_2d_channels(name: str, rows: list[list[float]], expected_len: int) -> np.ndarray:
    if not rows:
        raise ValueError(f"{name} must contain at least one channel")
    arrays: list[np.ndarray] = []
    for i, row in enumerate(rows):
        if not row:
            raise ValueError(f"{name}[{i}] must be a non-empty list of numbers")
        arr = np.asarray(row, dtype=np.float32).reshape(-1)
        if int(arr.size) != expected_len:
            raise ValueError(
                f"{name}[{i}] length is {int(arr.size)}, expected {expected_len}"
            )
        arrays.append(arr)
    return np.stack(arrays, axis=0)


def run_forecast(
    forecaster: Any,
    *,
    series: list[list[float]] | None = None,
    history: list[float] | None = None,
    horizon: int = 5,
    series_ids: list[str] | None = None,
    past_covariates: list[list[float]] | None = None,
    future_covariates: list[list[float]] | None = None,
) -> dict:
    try:
        if int(horizon) < 1:
            raise ValueError("horizon must be >= 1")
        target = resolve_series(series, history)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": str(exc)}

    n_series = int(target.shape[0])
    context_length = int(target.shape[1])

    if series_ids is not None and len(series_ids) != n_series:
        return {
            "status": "error",
            "error": f"series_ids length is {len(series_ids)}, expected {n_series}",
        }

    po = None
    pf = None
    try:
        if past_covariates:
            po = [_as_2d_channels("past_covariates", past_covariates, context_length)]
        if future_covariates:
            pf = [
                _as_2d_channels(
                    "future_covariates",
                    future_covariates,
                    context_length + int(horizon),
                )
            ]
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": str(exc)}

    # Official univariate path is a 1D context. Covariates and V>1 need (V, T).
    if n_series == 1 and po is None and pf is None:
        contexts: list[np.ndarray] = [target.reshape(-1)]
    else:
        contexts = [target]

    outputs = list(
        forecaster.predict_batch(
            contexts,
            horizon=int(horizon),
            past_only_covariates=po,
            past_future_covariates=pf,
            return_quantiles=True,
            use_symmetric_averaging=False,
        )
    )
    if not outputs:
        return {"status": "error", "error": "model returned no forecast"}

    first = outputs[0]
    forecast = np.asarray(first.forecast)
    quantiles = None if getattr(first, "quantiles", None) is None else np.asarray(first.quantiles)

    if forecast.ndim == 1:
        forecast = forecast.reshape(1, -1)
    elif forecast.ndim != 2:
        return {
            "status": "error",
            "error": f"unexpected forecast shape {forecast.shape}",
        }

    if quantiles is not None:
        if quantiles.ndim == 2:
            quantiles = quantiles.reshape(1, quantiles.shape[0], quantiles.shape[1])
        elif quantiles.ndim != 3:
            return {
                "status": "error",
                "error": f"unexpected quantiles shape {quantiles.shape}",
            }

    if forecast.shape[0] != n_series:
        return {
            "status": "error",
            "error": (
                f"model returned {forecast.shape[0]} series, expected {n_series}"
            ),
        }

    items = []
    for i in range(n_series):
        q_i = None if quantiles is None else quantiles[i]
        sid = series_ids[i] if series_ids is not None else f"series_{i}"
        items.append(
            {
                "id": sid,
                "forecast": forecast[i].astype(float).tolist(),
                "quantiles": serialize_quantiles(q_i),
            }
        )

    payload: dict[str, Any] = {
        "status": "success",
        "model": CHECKPOINT,
        "mode": "univariate" if n_series == 1 else "multivariate",
        "n_series": n_series,
        "context_length": context_length,
        "horizon": int(horizon),
        "series": items,
        "quantile_levels": QUANTILE_LEVELS,
        "license": LICENSE_NOTE,
    }
    if n_series == 1:
        payload["forecast"] = items[0]["forecast"]
        payload["quantiles"] = items[0]["quantiles"]
    return payload
