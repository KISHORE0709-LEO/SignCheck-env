import os
import sys
import time
import json
import requests
from typing import List, Optional
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
API_KEY      = os.getenv("HF_TOKEN")
BASE_URL     = os.getenv("SIGNCHECK_URL","http://localhost:7860")
MAX_STEPS    = 25

ACTIONS = [
    "SOUND_WARD_ALARM", "CALL_ATTENDING_DOCTOR", "CALL_ICU_SPECIALIST",
    "CALL_CODE_BLUE", "CHECK_EQUIPMENT", "CHECK_PATIENT_AIRWAY",
    "REPOSITION_PATIENT", "START_MANUAL_BAGGING", "ADJUST_OXYGEN_FLOW",
    "SILENCE_ALARM", "CHECK_IV_LINE", "ADMINISTER_EMERGENCY_MED",
    "WAIT_AND_MONITOR"
]

SYSTEM_PROMPT = """You are an ICU nurse AI acting as first responder.
A medical emergency is occurring and the doctor is not present.
You must stabilize the patient until the doctor arrives.

Each turn you will receive patient vitals, equipment status, and clinical notes.
You must respond with EXACTLY ONE action from this list, nothing else:

SOUND_WARD_ALARM       - Sound the ward alarm to alert nearby staff
CALL_ATTENDING_DOCTOR  - Call the attending doctor (routine escalation)
CALL_ICU_SPECIALIST    - Call the ICU specialist (equipment/ventilator issues)
CALL_CODE_BLUE         - Declare Code Blue (cardiac/life-threatening emergency)
CHECK_EQUIPMENT        - Check all equipment for faults or alarms
CHECK_PATIENT_AIRWAY   - Check patient airway patency
REPOSITION_PATIENT     - Reposition patient to improve breathing
START_MANUAL_BAGGING   - Start manual bag-valve-mask ventilation
ADJUST_OXYGEN_FLOW     - Adjust supplemental oxygen flow rate
SILENCE_ALARM          - Silence a non-critical alarm
CHECK_IV_LINE          - Check IV line for occlusion or infiltration
ADMINISTER_EMERGENCY_MED - Administer emergency medication (HIGH RISK)
WAIT_AND_MONITOR       - Wait and continue monitoring (use sparingly)

Rules:
- Respond with ONLY the action name, no punctuation, no explanation
- Read clinical notes carefully before acting
- Prioritize patient safety over speed
- Call the RIGHT person: Code Blue for cardiac, ICU Specialist for equipment, Attending for general
- Check before you treat — verify the cause before intervening
"""

# ── Logging ─────────────────────────────────────────────────
def log_start(task_id: str, task_name: str, model: str):
    print(f"[START] task={task_id} task_name={task_name} model={model} timestamp={int(time.time())}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} mean_reward={mean:.2f} rewards={rewards_str}", flush=True)

# ── Agent ────────────────────────────────────────────────────
def build_user_prompt(obs: dict, step: int, last_reward: float, history: List[str]) -> str:
    history_str = " → ".join(history[-5:]) if history else "None"
    return f"""=== STEP {step} ===
VITALS:
  SpO2:          {obs['spo2']}%
  Heart Rate:    {obs['heart_rate']} bpm
  BP:            {obs['bp_systolic']}/{obs['bp_diastolic']} mmHg
  Resp Rate:     {obs['resp_rate']} breaths/min
  Temperature:   {obs['temperature']}°C
  Consciousness: {obs['consciousness']}

EQUIPMENT: {json.dumps(obs['equipment_status'], indent=2)}
POWER:     {obs['power_status']}
DOCTOR ETA: {obs.get('doctor_eta', 'Not called yet')} steps

CLINICAL NOTES:
{obs['clinical_notes']}

LAST ACTION RESULT: {obs['last_action_feedback']}
LAST REWARD: {last_reward:.2f}
RECENT ACTIONS: {history_str}
PATIENT STATUS: {obs['patient_outcome']}

Choose your next action:"""

def get_model_action(client: OpenAI, obs: dict, step: int, last_reward: float, history: List[str]) -> str:
    prompt = build_user_prompt(obs, step, last_reward, history)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=20,
            temperature=0.0
        )
        raw = response.choices[0].message.content.strip().upper()
        # Clean up any extra text
        for action in ACTIONS:
            if action in raw:
                return action
        return "WAIT_AND_MONITOR"
    except Exception as e:
        return "WAIT_AND_MONITOR"

# ── Task Runner ──────────────────────────────────────────────
def run_task(client: OpenAI, task_id: int) -> dict:
    # Reset
    try:
        r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
        r.raise_for_status()
        reset_data = r.json()
    except Exception as e:
        print(f"[ERROR] Failed to reset task {task_id}: {e}", flush=True)
        return {"task_id": task_id, "score": 0.0, "success": False, "steps": 0}

    obs       = reset_data["observation"]
    task_name = reset_data["task_description"][:40].replace(" ", "_")

    log_start(str(task_id), task_name, MODEL_NAME)

    rewards        = []
    action_history = []
    last_reward    = 0.0
    done           = False
    step           = 0
    last_error     = None

    while not done and step < MAX_STEPS:
        step += 1
        action = get_model_action(client, obs, step, last_reward, action_history)

        try:
            r = requests.post(f"{BASE_URL}/step", json={"action": action}, timeout=30)
            r.raise_for_status()
            step_data = r.json()

            obs         = step_data["observation"]
            reward      = step_data["reward"]
            done        = step_data["done"]
            last_error  = step_data["info"].get("outcome") if done else None
            last_reward = reward

            rewards.append(reward)
            action_history.append(action)
            log_step(step, action, reward, done, None)

        except Exception as e:
            last_error = str(e)
            log_step(step, action, 0.0, True, last_error)
            break

    # Grade
    score = 0.0
    try:
        r = requests.post(f"{BASE_URL}/grade", timeout=30)
        r.raise_for_status()
        grade_data = r.json()
        score      = grade_data["final_score"]
        success    = grade_data["passed"]
    except Exception as e:
        success = False

    log_end(success, step, score, rewards)
    return {"task_id": task_id, "score": score, "success": success, "steps": step}

# ── Main ─────────────────────────────────────────────────────
def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print("\n" + "="*60, flush=True)
    print("SignCheck-env Baseline Inference", flush=True)
    print("="*60 + "\n", flush=True)

    results = []
    for task_id in [1, 2, 3]:
        print(f"\n{'='*60}", flush=True)
        result = run_task(client, task_id)
        results.append(result)
        time.sleep(1)

    # Summary
    print("\n" + "="*60, flush=True)
    print("FINAL RESULTS SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"{'Task':<8} {'Score':<10} {'Passed':<10} {'Steps'}", flush=True)
    print("-"*40, flush=True)
    for r in results:
        passed = "✅" if r["success"] else "❌"
        print(f"{r['task_id']:<8} {r['score']:<10.3f} {passed:<10} {r['steps']}", flush=True)

    avg = sum(r["score"] for r in results) / len(results)
    print(f"\nAverage Score: {avg:.3f}", flush=True)

if __name__ == "__main__":
    main()
