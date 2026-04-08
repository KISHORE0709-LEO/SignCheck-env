---
title: SignCheck
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# SignCheck-Env: ICU Emergency Response RL Environment

## The Problem
Many patients in hospitals can deteriorate rapidly or even tragically pass away during unforeseen emergencies such as power failures, sudden equipment malfunctions, or when there is a delayed response from a doctor. In an Intensive Care Unit (ICU), seconds matter. When a crisis occurs, quick and clinically precise decisions are absolutely critical to keep the patient alive until qualified human help arrives.

## The Solution
**SignCheck-Env** is an OpenEnv reinforcement learning environment where an AI agent acts as a life-saving first responder. The agent continuously monitors a simulated patient's vitals in real-time and performs critical emergency actions—from manually bagging a patient to checking IV lines—to stabilize them until an attending doctor or emergency team arrives on the scene.

## What Was Built
This project is an end-to-end simulation framework consisting of several interconnected systems:
- **Patient Vitals Simulator**: Generates realistic, constantly evolving physiological parameters like heart rate, oxygen saturation (SpO2), and blood pressure.
- **Emergency Scenario Engine**: Triggers complex events that the agent must learn to navigate (e.g., ventilator failure, sudden cardiac events, or power outages).
- **Reinforcement Learning Environment**: A state-machine tracking the episode timeline, the agent's actions, and immediate physiological consequences over time.
- **Grading System**: Automatically and deterministically calculates the agent's performance based on safety guidelines.
- **FastAPI Interface**: Provides easy-to-use REST API endpoints that allow any external code or agent to interact with the environment seamlessly.

## Key Functionalities
- **Realistic Patient Vital Drift**: Patient vitals naturally change based on underlying conditions and either stabilize or plummet based on the agent's interventions.
- **Multiple ICU Emergency Scenarios**: Tests the agent against a variety of distinct, highly stressful real-world crises.
- **Agent Actions & Escalation**: The agent can perform a wide variety of actions, such as checking equipment, adjusting oxygen flow, pushing emergency medication, and escalating safely to an ICU specialist or Code Blue team.
- **Automatic Grading**: Generates a detailed final score summarizing the agent's ability to maintain patient stability efficiently and correctly.

## Project Architecture
The system flow is highly modular and easy to understand:

**Agent (`inference.py`)** → **FastAPI Server (`server/main.py`)** → **Environment Engine (`server/env.py`)** → **Vitals Simulator (`server/vitals.py`)** → **Scenario Logic (`server/scenarios.py`)** → **Grader (`server/grader.py`)**

## How to Run Locally

Follow these exact steps to run the simulation and test the baseline agent on your machine.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/KISHORE0709-LEO/SignCheck-env.git
    cd SignCheck-env
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Start the environment server:**
    ```bash
    uvicorn server.main:app --port 7860
    ```

4.  **Explore the API Documentation:**
    Open [http://127.0.0.1:7860/docs](http://127.0.0.1:7860/docs) in your browser to view the interactive FastAPI Swagger interface.

5.  **Run the demo inference agent** (in a new terminal):
    ```bash
    python inference.py
    ```

## How to Test the Environment

You can easily test the environment manually using the interactive Swagger UI you just opened! Here is a simple workflow:

1. **Start a New Episode**: 
   Open the `POST /reset` endpoint panel, click **Try it out**, ensure the JSON request body says `{ "task_id": 1 }`, and click **Execute**. You will see the initial patient observation and context in the response body.
   
2. **Take an Action**: 
   Scroll to the `POST /step` endpoint, click **Try it out**, and enter an action in the request body, such as:
   ```json
   { "action": "CHECK_EQUIPMENT" }
   ```
   Click **Execute** and check the response to see how the patient's vitals changed and your immediate reward.
   
3. **Get Your Score**: 
   Once the episode completes (the `done` flag returns `true`), open the `POST /grade` endpoint, click **Try it out**, and hit **Execute** to see your final deterministic score based on the actions you took!
