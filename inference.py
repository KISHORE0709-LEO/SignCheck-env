import os
import time
import json
import asyncio
import requests
from typing import List, Dict, Any
from openai import OpenAI

API_BASE_URL = os.getenv('API_BASE_URL', 'https://api.openai.com/v1')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')
API_KEY = os.getenv('HF_TOKEN', '')

MAX_STEPS = 25
TEMPERATURE = 0.0
MAX_TOKENS = 512
SUCCESS_THRESHOLD = 0.6
BASE_URL = os.getenv('SIGNCHECK_URL', 'http://localhost:7860')

VALID_ACTIONS = [
    "SOUND_WARD_ALARM", "CALL_ATTENDING_DOCTOR", "CALL_ICU_SPECIALIST", 
    "CALL_CODE_BLUE", "CHECK_EQUIPMENT", "CHECK_PATIENT_AIRWAY", 
    "REPOSITION_PATIENT", "START_MANUAL_BAGGING", "ADJUST_OXYGEN_FLOW", 
    "SILENCE_ALARM", "CHECK_IV_LINE", "ADMINISTER_EMERGENCY_MED", 
    "WAIT_AND_MONITOR"
]

SYSTEM_PROMPT = """You are an AI first-responder agent monitoring an ICU patient's vitals. 
Your goal is to triage the patient appropriately, stabilize critical vitals, investigate anomalies, and escalate correctly. 
You must choose exactly ONE action per turn from the allowed Action enum. 
Respond ONLY with the exact text of the action. Do NOT provide any explanation or extra text."""

def log_start(task_id: int, task_name: str, model: str):
    unix_time = int(time.time())
    print(f"[START] task={task_id} task_name={task_name} model={model} timestamp={unix_time}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str):
    print(f"[STEP] step={step} action={action} reward={reward:.3f} done={done} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"[END] success={success} steps={steps} score={score:.3f} mean_reward={mean:.3f} rewards={rewards}", flush=True)

def build_user_prompt(obs_dict: Dict[str, Any], step: int, last_reward: float, action_history: List[str]) -> str:
    prompt = f"--- Step {step} ---\n"
    prompt += f"Observation:\n{json.dumps(obs_dict, indent=2)}\n\n"
    prompt += f"Action History:\n{action_history}\n\n"
    prompt += f"Last Reward: {last_reward:.3f}\n\n"
    prompt += "Allowed Actions: " + ", ".join(VALID_ACTIONS) + "\n\n"
    prompt += "Output strictly the chosen Action:"
    return prompt

def get_model_action(client: OpenAI, obs_dict: Dict[str, Any], step: int, last_reward: float, action_history: List[str]) -> str:
    user_prompt = build_user_prompt(obs_dict, step, last_reward, action_history)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        action_str = response.choices[0].message.content.strip()
        
        for act in VALID_ACTIONS:
            if act in action_str:
                return act
                
        return "WAIT_AND_MONITOR"
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "WAIT_AND_MONITOR"

def run_task(client: OpenAI, task_id: int) -> float:
    res = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
    if res.status_code != 200:
        print(f"Error resetting environment: {res.text}")
        return 0.0
        
    reset_data = res.json()
    obs = reset_data["observation"]
    task_desc = reset_data["task_description"]
    
    log_start(task_id, task_desc, MODEL_NAME)
    
    done = False
    step = 0
    last_reward = 0.0
    action_history = []
    rewards = []
    
    while not done and step < MAX_STEPS:
        step += 1
        
        action = get_model_action(client, obs, step, last_reward, action_history)
        
        step_res = requests.post(f"{BASE_URL}/step", json={"action": action})
        if step_res.status_code != 200:
            error_msg = str(step_res.text)
            log_step(step, action, 0.0, True, error_msg)
            break
            
        step_data = step_res.json()
        obs = step_data["observation"]
        last_reward = step_data.get("reward", 0.0)
        done = step_data.get("done", False)
        
        rewards.append(last_reward)
        action_history.append(action)
        
        log_step(step, action, last_reward, done, "")
        
    grade_res = requests.post(f"{BASE_URL}/grade")
    if grade_res.status_code == 200:
        grade_data = grade_res.json()
        final_score = grade_data["final_score"]
        passed = grade_data["passed"]
    else:
        final_score = sum(rewards) / max(len(rewards), 1)
        passed = False
        
    log_end(passed, step, final_score, rewards)
    return final_score

def main():
    for _ in range(5):
        try:
            health = requests.get(f"{BASE_URL}/health")
            if health.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
            
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY if API_KEY else "dummy_key")
    
    print(f"Starting Inference Evaluation against OpenEnv Server: {BASE_URL}")
    print("-" * 50)
    
    scores = {}
    for task_id in [1, 2, 3]:
        score = run_task(client, task_id)
        scores[task_id] = score
        
    print("\n" + "=" * 40)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 40)
    print(f"{'Task ID':<10} | {'Score':<10} | {'Status':<10}")
    print("-" * 40)
    for t_id, s in scores.items():
        status = "PASSED" if s >= SUCCESS_THRESHOLD else "FAILED"
        print(f"{t_id:<10} | {s:<10.3f} | {status:<10}")
    print("=" * 40)

if __name__ == "__main__":
    main()
