import os
import sys
import numpy as np
from fastmcp import FastMCP
from timesfm3 import TimesFM3Evaluator, ModelConfig
from huggingface_hub import login

# Cloud Authentication
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    sys.stderr.write("Authenticating with Hugging Face...\n")
    login(token=hf_token)
else:
    sys.stderr.write("WARNING: HF_TOKEN environment variable not found!\n")

mcp = FastMCP("TimesFM Server")

sys.stderr.write("Loading TimesFM-3 v3.0 model into memory...\n")
config = ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", device="cpu")
forecaster = TimesFM3Evaluator(config)
sys.stderr.write("Model ready!\n")

@mcp.tool()
def forecast_demand(history: list[float], horizon: int = 5) -> dict:
    """Predicts future time series values using TimesFM-3."""
    sys.stderr.write(f"Inference: {len(history)} points -> {horizon} steps\n")
    history_array = np.array(history, dtype=np.float32)
    
    results = forecaster.predict_batch(
        [history_array], horizon=horizon, return_quantiles=True, use_symmetric_averaging=False
    )
    
    first_result = next(results)
    predictions = first_result.forecast.tolist()
    
    return {"status": "success", "horizon": horizon, "forecast": predictions}

if __name__ == "__main__":
    # Network Shift: HTTP/SSE on port 7860
    mcp.run(transport="sse", host="0.0.0.0", port=7860)