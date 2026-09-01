# TimesFM-3 MCP Server

A lightweight Model Context Protocol (MCP) server that exposes Google's **TimesFM-3** zero-shot multivariate time-series forecasting model as a tool for AI agents.

## Features
- **Zero-Shot Forecasting:** Predicts future trends without needing task-specific fine-tuning.
- **FastMCP Integration:** Built with the latest FastMCP framework for seamless LLM agent orchestration via standard I/O (`stdio`).
- **Local Execution:** Runs entirely locally on your CPU using Google's official `timesfm3` package and PyTorch.

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/thenameisdevair/timesfm-mcp.git](https://github.com/thenameisdevair/timesfm-mcp.git)
   cd timesfm-mcp
   '''

2. Create and activate a virtual environment:

    '''bash
    python3 -m venv venv
    source venv/bin/activate
    '''

3. Install dependencies:

    '''Bash
    pip install -r requirements.txt
    pip install git+[https://github.com/google-research/timesfm.git](https://github.com/google-research/timesfm.git)

4. Authenticate with Hugging Face (required for downloading the gated google/timesfm-3.0-pytorch weights):

    '''Bash
    huggingface-cli login

5. Usage
    Run the client test script to verify inference:

    '''Bash
    python client.py