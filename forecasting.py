"""TimesFM-3 forecast helpers shared by the local and cloud MCP servers.

Shapes match Google's official TimesFM 3.0 API:

- Univariate context: 1D array of length T
- Multivariate context: 2D array of shape (V, T)
- past_only_covariates: (C_past, T)
- past_future_covariates: (C_future, T + H)
- Univariate output: forecast (H,), quantiles (H, 9)
- Multivariate output: forecast (V, H), quantiles (V, H, 9)

Optional calendar labels (start + freq, or a history timestamp list) are
applied after inference. They are never passed to TimesFM-3.
"""

from __future__ import annotations

from calendar import monthrange
from datetime import datetime, timedelta
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

FREQ_ALIASES = {
    "h": "H",
    "hour": "H",
    "hours": "H",
    "d": "D",
    "day": "D",
    "days": "D",
    "w": "W",
    "week": "W",
    "weeks": "W",
    "m": "M",
    "month": "M",
    "months": "M",
}


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


def _is_date_only(value: str) -> bool:
    s = str(value).strip()
    return len(s) == 10 and s[4] == "-" and s[7] == "-"


def parse_timestamp(value: str) -> datetime:
    s = str(value).strip()
    if not s:
        raise ValueError("timestamp is empty")
    if _is_date_only(s):
        return datetime.fromisoformat(s)
    s = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


def _format_timestamp(dt: datetime, date_only: bool) -> str:
    if date_only:
        return dt.date().isoformat()
    return dt.replace(microsecond=0).isoformat()


def _add_months(dt: datetime, n: int) -> datetime:
    month = dt.month - 1 + n
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def add_steps(dt: datetime, freq: str, n: int) -> datetime:
    if n == 0:
        return dt
    if freq == "H":
        return dt + timedelta(hours=n)
    if freq == "D":
        return dt + timedelta(days=n)
    if freq == "W":
        return dt + timedelta(weeks=n)
    if freq == "M":
        return _add_months(dt, n)
    raise ValueError(f"freq must be one of H, D, W, M (got {freq!r})")


def _normalize_freq(freq: str) -> str:
    key = FREQ_ALIASES.get(str(freq).strip().lower())
    if key is None:
        raise ValueError(f"freq must be one of H, D, W, M (got {freq!r})")
    return key


def _freq_from_delta(delta: timedelta) -> str | None:
    seconds = int(delta.total_seconds())
    if seconds == 3600:
        return "H"
    if seconds == 86400:
        return "D"
    if seconds == 86400 * 7:
        return "W"
    return None


def resolve_calendar(
    *,
    context_length: int,
    horizon: int,
    start: str | None = None,
    freq: str | None = None,
    timestamps: list[str] | None = None,
) -> dict[str, Any] | None:
    has_start = start is not None and str(start).strip() != ""
    has_freq = freq is not None and str(freq).strip() != ""
    has_ts = timestamps is not None

    if not has_start and not has_freq and not has_ts:
        return None
    if has_ts and (has_start or has_freq):
        raise ValueError("pass timestamps or start+freq, not both")
    if has_ts:
        return _calendar_from_timestamps(timestamps, context_length, horizon)
    if has_start ^ has_freq:
        raise ValueError("start and freq must be provided together")
    return _calendar_from_start_freq(str(start), str(freq), context_length, horizon)


def _calendar_from_start_freq(
    start: str, freq: str, context_length: int, horizon: int
) -> dict[str, Any]:
    freq_key = _normalize_freq(freq)
    origin = parse_timestamp(start)
    date_only = _is_date_only(start) and freq_key != "H"
    history_end = add_steps(origin, freq_key, context_length - 1)
    forecast_timestamps = [
        _format_timestamp(add_steps(origin, freq_key, context_length + i), date_only)
        for i in range(horizon)
    ]
    return {
        "freq": freq_key,
        "history_end": _format_timestamp(history_end, date_only),
        "forecast_timestamps": forecast_timestamps,
    }


def _calendar_from_timestamps(
    timestamps: list[str], context_length: int, horizon: int
) -> dict[str, Any]:
    if len(timestamps) != context_length:
        raise ValueError(
            f"timestamps length is {len(timestamps)}, expected {context_length}"
        )
    parsed = [parse_timestamp(value) for value in timestamps]
    if any(later <= earlier for earlier, later in zip(parsed, parsed[1:])):
        raise ValueError("timestamps must be strictly increasing")
    if context_length < 2:
        raise ValueError("need at least two timestamps to check spacing, or pass start and freq")
    deltas = [later - earlier for earlier, later in zip(parsed, parsed[1:])]
    if any(delta != deltas[0] for delta in deltas[1:]):
        raise ValueError(
            "timestamps are not strictly regular; missing observations are not filled"
        )
    step = deltas[0]
    date_only = all(_is_date_only(value) for value in timestamps)
    last = parsed[-1]
    forecast_timestamps = [
        _format_timestamp(last + step * (i + 1), date_only) for i in range(horizon)
    ]
    inferred = _freq_from_delta(step)
    payload = {
        "history_end": _format_timestamp(last, date_only),
        "forecast_timestamps": forecast_timestamps,
    }
    if inferred is not None:
        payload["freq"] = inferred
    return payload


def run_forecast(
    forecaster: Any,
    *,
    series: list[list[float]] | None = None,
    history: list[float] | None = None,
    horizon: int = 5,
    series_ids: list[str] | None = None,
    past_covariates: list[list[float]] | None = None,
    future_covariates: list[list[float]] | None = None,
    start: str | None = None,
    freq: str | None = None,
    timestamps: list[str] | None = None,
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
        calendar = resolve_calendar(
            context_length=context_length,
            horizon=int(horizon),
            start=start,
            freq=freq,
            timestamps=timestamps,
        )
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

    stamp = None if calendar is None else calendar["forecast_timestamps"]
    items = []
    for i in range(n_series):
        q_i = None if quantiles is None else quantiles[i]
        sid = series_ids[i] if series_ids is not None else f"series_{i}"
        item: dict[str, Any] = {
            "id": sid,
            "forecast": forecast[i].astype(float).tolist(),
            "quantiles": serialize_quantiles(q_i),
        }
        if stamp is not None:
            item["timestamps"] = stamp
        items.append(item)

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
    if calendar is not None:
        if calendar.get("freq") is not None:
            payload["freq"] = calendar["freq"]
        payload["history_end"] = calendar["history_end"]
        if n_series == 1:
            payload["timestamps"] = stamp
    if n_series == 1:
        payload["forecast"] = items[0]["forecast"]
        payload["quantiles"] = items[0]["quantiles"]
    return payload
