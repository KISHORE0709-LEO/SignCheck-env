# Quick Start Guide

Welcome to the SignCheck-env benchmark! If you are new here and just want to run the environment or see how an AI interacts with it, follow these steps:

### 1. Install Requirements
You need Python 3.11+. Install the required libraries via terminal:
```bash
pip install -r requirements.txt
```

### 2. Boot the API Server
Start the simulation backend. It runs using FastAPI on port 7860.
```bash
uvicorn server.main:app --port 7860
```
*You can view the interactive documentation at `http://localhost:7860/docs`.*

### 3. Open the UI Dashboard (Optional, but recommended)
Open a new terminal window and run the monitoring dashboard to visualize the patient:
```bash
streamlit run dashboard/dashboard.py
```

### 4. Run an AI Agent
Open a third terminal window. Provide your API key and let an LLM attempt to triage the patient:
```bash
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
python agent/inference.py
```

Watch the dashboard as the agent begins making medical decisions! Alternatively, if you want a non-AI heuristic runner, use `python scripts/playground.py`.
