# TimesFM-3 MCP Server

MCP server that exposes Google's **TimesFM-3** zero-shot time-series foundation model to AI agents (Claude Desktop, Claude Code, Cursor, and any MCP client).

Give an agent a history series. Get a median forecast plus nine quantile bands (`q10`–`q90`).

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

```json
{
  "history": [10.5, 12.1, 14.8, 15.2, 18.0],
  "horizon": 5
}
```

```json
{
  "status": "success",
  "model": "google/timesfm-3.0-pytorch",
  "context_length": 5,
  "horizon": 5,
  "forecast": [18.4, 19.1, 19.8, 20.2, 20.9],
  "quantiles": {
    "q10": [16.1, 16.4, 16.8, 17.0, 17.3],
    "q50": [18.4, 19.1, 19.8, 20.2, 20.9],
    "q90": [21.0, 22.1, 23.0, 23.8, 24.6]
  },
  "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
  "license": "TimesFM-3 pretrained weights are non-commercial / non-production..."
}
```

`forecast` is the median (q50). `quantiles` are the nine official TimesFM-3 heads, 10th through 90th percentile. Numbers above are illustrative.

`forecast_demand` still exists as a deprecated alias so older configs do not break.

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

Then ask: *Forecast the next 8 steps from this series and tell me the q10 / q50 / q90 range.*

## Cloud (Hugging Face Space)

`cloud/` is a Docker image that serves the same tool over HTTP/SSE on port 7860.

1. Create a **Docker** Space.
2. Set the build context to the `cloud/` folder, or copy `cloud/Dockerfile`, `cloud/app.py`, and `cloud/requirements.txt` to the Space root.
3. Add Space secret `HF_TOKEN` (a Hugging Face token that can download `google/timesfm-3.0-pytorch`).
4. Keep the Space marked research / non-commercial. The weights license does not allow a paid hosted forecast API.

## Status

Implemented now:

- Univariate `forecast` with point + 9 quantiles
- Local stdio + cloud SSE
- Explicit license banner on the tool and in this README

Not in this release (on purpose):

- Native multivariate targets
- Past-only / past-future covariates
- Commercial inference backend (TimesFM 2.5 or BigQuery)

Those are the next product steps. This cut is the shareable TimesFM-3 MCP surface.
