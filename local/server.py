import sys
import numpy as np
from fastmcp import FastMCP
from timesfm3 import TimesFM3Evaluator, ModelConfig

# Initialize the server
mcp = FastMCP("TimesFM Server")

sys.stderr.write("Loading TimesFM-3 v3.0 model into memory...\n")

# Use checkpoint_path instead of repo_id
config = ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", device="cpu")
forecaster = TimesFM3Evaluator(config)

sys.stderr.write("Model ready!\n")

@mcp.tool()
def forecast_demand(history: list[float], horizon: int = 5) -> dict:
    """
    Predicts future time series values using TimesFM-3.
    """
    sys.stderr.write(f"Inference: {len(history)} points -> {horizon} steps\n")
    
    # 1. Convert to float32 NumPy array
    history_array = np.array(history, dtype=np.float32)
    
    # 2. Run the V3 prediction generator
    results = forecaster.predict_batch(
        [history_array], 
        horizon=horizon, 
        return_quantiles=True, 
        use_symmetric_averaging=False
    )
    
    # 3. Get the first result from the generator
    first_result = next(results)
    
    # 4. Extract the 'forecast' attribute and convert the NumPy array to a standard Python list
    predictions = first_result.forecast.tolist()
    
    return {
        "status": "success",
        "horizon": horizon,
        "forecast": predictions
    }

if __name__ == "__main__":
    mcp.run()