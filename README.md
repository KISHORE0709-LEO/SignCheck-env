# SignCheck-Env: ICU Emergency Response RL Environment

## Overview

This is an **OpenEnv** reinforcement learning environment where an AI agent stabilizes a patient during hospital emergencies when a doctor is not present. The environment rigorously evaluates the capability of an autonomous agent to act as a first responder in high-stakes, life-or-death intensive care situations.

## Key Idea

The environment simulates dynamically degrading patient vitals in an Intensive Care Unit (ICU) setting. To succeed, the agent must continuously monitor these physiological parameters and take immediate, correct actions to prevent critical failure. Available actions include checking medical equipment, adjusting oxygen delivery systems, manually bagging the patient, or escalating the situation to a Code Blue team, ensuring patient stability until human help arrives.

## Environment Tasks

The environment evaluates the agent across three primary emergency scenarios:

*   **Task 1: Power Failure**: A sudden loss of primary power compromises essential, life-supporting medical equipment. The agent is forced to rapidly secure backup support and stabilize the patient's breathing manually.
*   **Task 2: Ventilator Malfunction**: The mechanical ventilator abruptly fails to deliver necessary oxygen. The agent must quickly recognize the mechanical failure through the degrading vitals and intervene accordingly.
*   **Task 3: Sudden Cardiac Event**: The patient experiences a drastic and rapid cardiac decline. The agent must recognize the severity of the drop and properly escalate the response via emergency protocols.

## Project Architecture

The simulation is built with a decoupled, container-friendly architecture. The system flow operates as follows:

**Agent** (`inference.py`) → **FastAPI Server** (`server/main.py`) → **Environment** (`server/env.py`) → **Vitals Simulator** (`server/vitals.py`) → **Scenario Logic** (`server/scenarios.py`) → **Grader** (`server/grader.py`).

## How to Run Locally

Follow these simple steps to run the simulation and test the baseline agent on your machine.

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

4.  **Run the demo inference agent (in a new terminal):**
    ```bash
    python inference.py
    ```

## API Endpoints

The environment is exposed via a RESTful FastAPI interface, adhering to typical reinforcement learning standards:

*   `GET /reset`: Initializes or resets the environment to the default starting state and returns the Initial Observation.
*   `POST /step`: Accepts an action from the agent, advances the simulation by one time step, and returns the updated `Observation`, `Reward`, `Done` flag, and `Info`.
*   `GET /state`: Returns the current physiological state of the patient without advancing the simulation clock.
*   `GET /grade`: Returns the comprehensive deterministic evaluation of the agent's performance across the scenario.

## Hackathon Context

This project is built using the **OpenEnv framework**. It aims to provide a robust benchmark to evaluate an agent's ability to interpret complex temporal data and respond to critical ICU scenarios in real-time. It serves as a proof-of-concept for deploying AI safely in acute medical environments.
