# SignCheck-Env

ICU patient triage constitutes one of the most stressful, time-critical, and consequential domains in modern medicine. When physiological collapse occurs, there is an inherent delay—a "critical gap"—between the onset of an emergency and the arrival of a specialist physician. AI agents evaluating real-time vital streams must act as first-responders during this window, making sequence-critical decisions balancing life-saving interventions with the risk of causing iatrogenic harm. By embedding clinical language reasoning, stochastic event dynamics, and cascading false-alarms into an MDP framework, SignCheck-Env serves as a uniquely rigorous safety-critical benchmark.

## Environment Description

SignCheck-Env runs as a centralized REST API defining three increasingly difficult medical scenarios. It subjects agents to an inherently unstable environment characterized by continuous vital sign degradation, nonlinear cascading consequences, and overlapping equipment states. The agent must stabilize the patient without prematurely expending time and must trigger the correct hospital escalation protocol matched perfectly to the respective acuity.

## Observation Space

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
| `time_elapsed` | int | Step delta counter | - |
| `time_since_last_vitals_check` | int | Steps since checking patient physically | - |
| `doctor_eta` | int | Steps till escalation arrives | - |
| `clinical_notes` | str | Flowing text descriptions | - |
| `last_action_feedback` | str | Evaluator prompt string | - |
| `step_count` | int | Raw sequence index | - |
| `patient_outcome` | str | Tracking variable | "stable" |

## Action Space

| Action | Description | Risk Level |
|--------|-------------|------------|
| `SOUND_WARD_ALARM` | Alert local nurses | Low |
| `CALL_ATTENDING_DOCTOR` | Routine physican escalation | Low |
| `CALL_ICU_SPECIALIST` | High-priority intensive alert | Medium |
| `CALL_CODE_BLUE` | Hospital-wide cardiac emergency | High |
| `CHECK_EQUIPMENT` | Inspect machine states (resolves false alarms) | Low |
| `CHECK_PATIENT_AIRWAY` | Ensure physical airflow | Low |
| `REPOSITION_PATIENT` | Adjust posture | Low |
| `START_MANUAL_BAGGING` | Take over ventilation | High |
| `ADJUST_OXYGEN_FLOW` | Regulate standard oxygen | Medium |
| `SILENCE_ALARM` | Mute telemetry noise | Low |
| `CHECK_IV_LINE` | Verify fluid/med delivery | Low |
| `ADMINISTER_EMERGENCY_MED` | Pharmaceutical intervention | High |
| `WAIT_AND_MONITOR` | Do nothing this step | Low |

## Task Descriptions

### Task 1: Power Failure
- **Difficulty:** Easy
- **Scenario:** The hospital wing experiences a power failure. Equipment switches to battery power but monitors drop offline. Vital signs will degrade slowly.
- **Win Condition:** Re-stabilize passive oxygen and call the Attending Doctor before vitals decay beyond recovery.
- **Lose Condition:** 30 steps elapsed or patient reaches 2+ critical vitals.

### Task 2: Equipment Malfunction
- **Difficulty:** Medium
- **Scenario:** A ventilator occlusion alarm sounds while patient vitals visibly worsen. 30% of the time, this alarm is entirely fake.
- **Win Condition:** Manually check equipment to isolate false alarms vs real crashes, perform manual bagging if true obstruction, and escalate to ICU Specialist.
- **Lose Condition:** Administering invasive bagging during a false alarm un-checked, 30 steps elapsed, or patient collapses.

### Task 3: Sudden Cardiac Event
- **Difficulty:** Hard
- **Scenario:** Immediate erratic heart rhythm accompanied by rapid multi-system collapse (BP loss, SpO2 loss).
- **Win Condition:** Bypass standard channels and call Code Blue within the first 4 steps, then administer emergency pharmacology correctly.
- **Lose Condition:** Misidentifying the crisis and calling the Attending Doctor (fatal delay), 25 steps elapsed, or patient drops to unresponsive with critical vitals.

## Reward Function

The episode returns a composite final score (0.0 - 1.0) weighted across four pillars:
- **Survival (40% Weight):** Earns 1.0 if patient is saved and doctor arrives, 0.5 if stable at time-out, 0.0 if deceased.
- **Vital Stability (30% Weight):** Computes the percentage of vital signs held within the normal range during every step, averaged across the episode. 
- **Escalation Quality (20% Weight):** Rewards strictly mapping the correct escalation code to the task acuity. Timing penalties applied if the code is triggered too late.
- **Step Efficiency (10% Weight):** Mathematical bias rewarding environments stabilized in fewer steps. `Ratio = 1.0 - (step_count/max_steps)`.

## Setup & Usage

To spin the environment up using Docker:

```bash
docker build -t signcheck-env .
docker run -p 7860:7860 signcheck-env
```

Once running locally on port 7860, run inference:

```bash
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
python inference.py
```

## Baseline Scores

| Task | Score | Passed |
|------|-------|--------|
| Task 1 | TBD | TBD |
| Task 2 | TBD | TBD |
| Task 3 | TBD | TBD |

## Citation

*(To be updated when published)*
