# TimesFM-3 MCP Server

MCP server that exposes Google's **TimesFM-3** zero-shot time-series foundation model to AI agents (Claude Desktop, Claude Code, Cursor, and any MCP client).

Pass one series or several related series, plus optional past / known-future covariates. Get a median forecast plus nine quantile bands (`q10`–`q90`) per series.

**Repo:** https://github.com/thenameisdevair/timesfm3-mcp

## License — read this first

| Piece | License | What you can do |
| --- | --- | --- |
| This MCP wrapper | Apache-2.0 | Use, modify, ship the *server code* |
| TimesFM-3 **weights** (`google/timesfm-3.0-pytorch`) | `timesfm-non-commercial-license-v1.0` | Research, evaluation, non-production experiments only |

Do **not** put this server on a paid product, customer workflow, or production planner while it loads the public TimesFM-3 checkpoint. Google's commercial path is BigQuery / AlloyDB `AI.FORECAST`.

The `forecast` tool repeats this warning in every response.

## What the agent gets

Tool: `forecast`

Univariate (also accepted as `history` for older clients):

```json
{
  "series": [[10.5, 12.1, 14.8, 15.2, 18.0]],
  "horizon": 5
}
```

Joint multivariate with a known-future promo flag (`future_covariates` length is `T + horizon`):

```json
{
  "series": [
    [10, 11, 13, 12, 14, 16, 15, 17],
    [20, 21, 22, 24, 23, 26, 25, 28]
  ],
  "series_ids": ["sku_a", "sku_b"],
  "future_covariates": [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]],
  "horizon": 3
}
```

Response (multivariate):

```json
{
  "status": "success",
  "model": "google/timesfm-3.0-pytorch",
  "mode": "multivariate",
  "n_series": 2,
  "context_length": 8,
  "horizon": 3,
  "series": [
    {
      "id": "sku_a",
      "forecast": [17.2, 17.8, 18.1],
      "quantiles": {
        "q10": [15.0, 15.4, 15.7],
        "q50": [17.2, 17.8, 18.1],
        "q90": [20.1, 21.0, 21.6]
      }
    }
  ],
  "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
  "license": "TimesFM-3 pretrained weights are non-commercial / non-production..."
}
```

Numbers above are illustrative. Univariate calls also include top-level `forecast` and `quantiles` so older clients keep working.

Shape rules (same as TimesFM-3):

- All target rows have length `T`
- `past_covariates` each have length `T`
- `future_covariates` each have length `T + horizon`

`forecast_demand(history, horizon)` remains a deprecated alias for a single series.

## Local setup

Weights are gated on Hugging Face. Accept the model terms, then log in.

```bash
git clone https://github.com/thenameisdevair/timesfm3-mcp.git
cd timesfm3-mcp

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r local/requirements.txt
pip install git+https://github.com/google-research/timesfm.git

huggingface-cli login
```

Shape/validation tests (no weight download):

```bash
python -m unittest tests.test_forecasting -v
```

Smoke-test inference (loads the 330M checkpoint; first run downloads weights):

```bash
cd local
python client.py
```

Start the stdio server the same way an MCP client will:

```bash
cd local
python server.py
```

RAM: plan for roughly 16GB+ on CPU. A GPU is optional; the server uses CUDA when PyTorch sees it.

## Connect an agent

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "timesfm3": {
      "command": "python",
      "args": ["/ABS/PATH/timesfm3-mcp/local/server.py"]
    }
  }
}
```

Use the interpreter from the venv that has `fastmcp`, `torch`, and `timesfm` installed. Example:

```json
"command": "/ABS/PATH/timesfm3-mcp/venv/bin/python"
```

### Claude Code

```bash
claude mcp add timesfm3 -- /ABS/PATH/timesfm3-mcp/venv/bin/python /ABS/PATH/timesfm3-mcp/local/server.py
```

### Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "timesfm3": {
      "command": "/ABS/PATH/timesfm3-mcp/venv/bin/python",
      "args": ["/ABS/PATH/timesfm3-mcp/local/server.py"]
    }
  }
}
```

Then ask: *Jointly forecast sku_a and sku_b for 8 steps, with this promo flag as a future covariate, and give q10 / q50 / q90.*

## Cloud (Hugging Face Space)

`cloud/` is a Docker image that serves the same tool over HTTP/SSE on port 7860.

1. Create a **Docker** Space.
2. Set the build context to the `cloud/` folder (it includes `forecasting.py`), or copy `cloud/Dockerfile`, `cloud/app.py`, `cloud/forecasting.py`, and `cloud/requirements.txt` to the Space root.
3. Add Space secret `HF_TOKEN` (a Hugging Face token that can download `google/timesfm-3.0-pytorch`).
4. Keep the Space marked research / non-commercial. The weights license does not allow a paid hosted forecast API.

## Status

Implemented now:

- Univariate and joint multivariate `forecast` with point + 9 quantiles
- Past-only and past-future covariates
- Local stdio + cloud SSE
- Explicit license banner on the tool and in this README

Not in this release:

- Commercial inference backend (TimesFM 2.5 or BigQuery)
