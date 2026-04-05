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
- **`Dockerfile`**: Packages the environment perfectly so judges can run it without setup failures.

---

## 🚀 How to Run (Quick Start)

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Start the core simulation server**
```bash
uvicorn server.main:app --port 7860
```

**3. Open the API Documentation**
Navigate to `http://localhost:7860/docs` in your browser to test endpoints manually.

**4. Run the Visual Dashboard** (Open a new terminal)
```bash
streamlit run dashboard/dashboard.py
```

**5. Run the AI Agent** (Open a new terminal)
```bash
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
python agent/inference.py
```

**6. Run the Debug Tests**
```bash
# Heuristic terminal runner
python scripts/playground.py --task 1

# Environment smoke test
python scripts/test_env.py
```

---

## 🔬 Deep Dive: Environment Mechanics

### Observation Space

| Field | Type | Description | Normal Range |
|-------|------|-------------|--------------|
| `spo2` | float | Oxygen saturation percentage | 95.0 - 100.0 |
| `heart_rate` | int | Heart beats per minute | 60 - 100 |
| `bp_systolic` | int | Blood pressure systolic | 110 - 130 |
| `bp_diastolic` | int | Blood pressure diastolic | 70 - 85 |
| `resp_rate` | int | Respiratory rate | 12 - 20 |
| `temperature` | float | Body temperature (C) | 36.5 - 37.5 |
| `consciousness` | str | AVPU Scale (Alert/Voice/Pain/Unresponsive) | "Alert" |
| `equipment_status` | dict | Status of monitor, vent, IV | "normal" |
| `power_status` | str | Macro power state of ward | "normal" |
| `clinical_notes` | str | Flowing text descriptions of the patient | - |

### Action Space

| Action | Description | Risk Level |
|--------|-------------|------------|
| `SOUND_WARD_ALARM` | Alert local nurses | Low |
| `CALL_ATTENDING_DOCTOR` | Routine physician escalation | Low |
| `CALL_ICU_SPECIALIST` | High-priority intensive alert | Medium |
| `CALL_CODE_BLUE` | Hospital-wide cardiac emergency | High |
| `CHECK_EQUIPMENT` | Inspect machine states (resolves false alarms) | Low |
| `CHECK_PATIENT_AIRWAY` | Ensure physical airflow | Low |
| `REPOSITION_PATIENT` | Adjust posture | Low |
| `START_MANUAL_BAGGING` | Take over ventilation manually | High |
| `ADJUST_OXYGEN_FLOW` | Regulate standard oxygen | Medium |
| `SILENCE_ALARM` | Mute telemetry noise | Low |
| `CHECK_IV_LINE` | Verify fluid/med delivery | Low |
| `ADMINISTER_EMERGENCY_MED` | Pharmaceutical intervention | High |
| `WAIT_AND_MONITOR` | Do nothing this step | Low |

### Task Descriptions

**Task 1: Power Failure (Easy)**
- **Scenario:** The hospital wing experiences a power failure. Equipment switches to battery power but monitors drop offline. Vital signs will degrade slowly.
- **Win Condition:** Re-stabilize passive oxygen and call the Attending Doctor before vitals decay beyond recovery.

**Task 2: Equipment Malfunction (Medium)**
- **Scenario:** A ventilator occlusion alarm sounds while patient vitals visibly worsen. 30% of the time, this alarm is entirely fake.
- **Lose Condition:** Administering invasive bagging during a false alarm un-checked.

**Task 3: Sudden Cardiac Event (Hard)**
- **Scenario:** Immediate erratic heart rhythm accompanied by rapid multi-system collapse (BP loss, SpO2 loss).
- **Win Condition:** Bypass standard channels and call Code Blue within the first 4 steps, then administer emergency pharmacology correctly.

### Reward Function
The episode returns a composite final score (0.0 - 1.0) weighted across four pillars:
- **Survival (40%)**: Earns 1.0 if patient is saved and doctor arrives, 0.5 if stable at time-out, 0.0 if deceased.
- **Vital Stability (30%)**: Percentage of vital signs held within the normal range during every step.
- **Escalation Quality (20%)**: Rewards strictly mapping the correct escalation code to the task acuity.
- **Step Efficiency (10%)**: Mathematical bias rewarding environments stabilized in fewer steps.

---

### Why Docker & HuggingFace?
- **Docker** is used strictly to package the environment so judges and public participants can run it easily offline on any machine without hitting Python dependency issues.
- **Hugging Face Spaces** allows us to host and share the environment publicly over the internet for robust OpenEnv-compliant remote evaluation.
