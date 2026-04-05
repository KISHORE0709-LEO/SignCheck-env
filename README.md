# SignCheck-Env
**ICU Patient Monitoring Simulation**

SignCheck-Env evaluates an AI's ability to act as a first-responder in the "critical gap"—the stressful time between the onset of a life-threatening medical emergency and the arrival of a specialized physician. 

### Problem Statement
When a patient suddenly crashes in a hospital (e.g., Sudden Cardiac Arrest, Ventilator Failures), nurses or on-call staff must sequence life-saving interventions flawlessly while fighting against extreme time pressure and false alarms. AI agents must prove they can reason across noisy data streams without causing fatal iatrogenic harm.

### What This Project Does
This project provides a robust, OpenEnv-compliant reinforcement learning benchmark encompassing:
- **Patient Simulation**: Realistic, continuous vital sign degradation modeled across continuous steps.
- **AI Decision-Making**: Agents must distinguish true clinical failures from sensor occlusion traps mapped inside complex text-based clinical notes.
- **Step-by-step environment**: State-Action-Reward cycles updating every step.
- **Scoring System**: Four-pillar grading evaluating long-term stability, correct escalation choices, structural efficiency, and overall survival rates.

## System Architecture

```text
User / LLM Agent 
       ↓ (Issues 1 of 13 Actions)
   FastAPI Server
       ↓
 Environment Engine (server/env.py)
       ↓ (Calculates Drift, Cascades, Interventions)
    Vitals Simulation
       ↓
  Step Reward & Outcome Output
```

## Project Components
- **`server/`**: The core physics environment logic (vitals, drift math, FastAPI routes).
- **`agent/`**: Baseline AI agent logic making decisions against the server.
- **`dashboard/`**: The Streamlit user-interface visualization showing live telemetry graphs.
- **`scripts/`**: Testing tools and local heuristic runners.
- **`configs/openenv.yaml`**: OpenEnv configuration limits and metadata definitions.
- **`Dockerfile`**: Used to package the environment perfectly so judges can run it without setup failures.

## How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start server**
   ```bash
   uvicorn server.main:app --port 7860
   ```

3. **Open API docs**
   Navigate to `http://localhost:7860/docs` in your browser.

4. **Run agent** (Run in a new terminal)
   ```bash
   export MODEL_NAME="gpt-4o-mini"
   export HF_TOKEN="your-api-key"
   python agent/inference.py
   ```

5. **Run dashboard** (Run in a new terminal)
   ```bash
   streamlit run dashboard/dashboard.py
   ```

6. **Run test**
   ```bash
   python scripts/test_env.py
   ```

## Example Workflow
1. Reset environment via `/reset` to select a patient.
2. Observe vitals, take medical actions via `/step`.
3. Watch vitals change interactively via drift math or successful interventions.
4. Call `/grade` to stop the episode and retrieve the final numeric score breakdown!

### Why Docker is Used
Docker is used strictly to package the environment so judges and public participants can run it easily offline on any machine without hitting Python versioning or setup dependency issues.

### Why HuggingFace is Used
Hugging Face Spaces allows us to host and share the environment publicly over the internet for robust OpenEnv-compliant remote evaluation.
