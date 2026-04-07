# SignCheck-env

## Overview
SignCheck-env is an ICU emergency response simulation environment. An AI agent is tasked with acting as a first-responder in the "critical gap" between the onset of a life-threatening medical emergency and the arrival of a specialized physician. The AI must monitor vital signs, interpret clinical text notes, and choose the correct sequence of interventions to stabilize a deteriorating patient.

## Tasks
The environment encompasses three scenarios of varying difficulty:
1. **Equipment Failure (Easy):** A ventilator occlusion alarm sounds while patient vitals worsen. The agent must distinguish true clinical failures from sensor errors.
2. **Oxygen Drop (Medium):** The ward loses power, and backup systems engage. Vitals degrade passively, requiring the agent to re-stabilize oxygen levels quickly safely. 
3. **Cardiac Deterioration (Hard):** The patient enters sudden erratic cardiac rhythm accompanied by rapid multi-system collapse. The agent must rapidly trigger a hospital-wide emergency code while navigating complex traps.

## Agent Interaction
Agents interact with the environment via a standard REST API loop mimicking typical RL interactions:
* **/reset**: Initializes the environment with a specific `task_id` and returns the starting physiological baselines (vitals and clinical notes).
* **/step**: Accepts an `action` from the action space, simulating one time-step of vital degradation and applying the action's effects. Returns the new observation and step reward.
* **/grade**: Evaluates the entire trajectory and computes a final score based on Survival, Vital Stability, Escalation Quality, and Efficiency.

## Architecture
The repository strictly adheres to a decentralized client-server architecture:
* **FastAPI Server (`server/main.py`)**: Hosts the API endpoints and holds the master state of the simulation.
* **Vitals Simulator (`server/vitals.py` & `server/env.py`)**: Calculates deterministic vital degradation, physiological responses to actions, and localized noise.
* **Scenarios (`server/scenarios.py`)**: Stores the task baselines and rulesets.
* **Grader (`server/grader.py`)**: The OpenEnv deterministic evaluation system that grades AI performance based on four distinct stability pillars.

## Local run instructions

```bash
pip install -r requirements.txt
uvicorn server.main:app --port 7860
python inference.py
```
