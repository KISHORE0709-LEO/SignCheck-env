"""Pre-submission validation script for SignCheck-env."""
import os
import sys
import json
import requests
import subprocess
import time

BASE_URL = os.getenv("SIGNCHECK_URL", "http://localhost:7860")
PASS = "✅"
FAIL = "❌"
errors = []

def check(label, ok, detail=""):
    status = PASS if ok else FAIL
    print(f"  {status} {label}" + (f": {detail}" if detail else ""))
    if not ok:
        errors.append(label)

print("\n=== SignCheck-env Pre-Submission Validator ===\n")

# 1. Env vars
print("[1] Environment Variables")
check("API_BASE_URL set", bool(os.getenv("API_BASE_URL")))
check("MODEL_NAME set",   bool(os.getenv("MODEL_NAME")))
check("HF_TOKEN set",     bool(os.getenv("HF_TOKEN")))

# 2. Required files
print("\n[2] Required Files")
for f in ["inference.py", "openenv.yaml", "Dockerfile", "requirements.txt"]:
    check(f, os.path.isfile(f))

# 3. openenv.yaml
print("\n[3] openenv.yaml")
try:
    import yaml
    with open("openenv.yaml") as f:
        spec = yaml.safe_load(f)
    check("has tasks",            "tasks" in spec and len(spec["tasks"]) >= 3)
    check("has observation_space","observation_space" in spec)
    check("has action_space",     "action_space" in spec)
    check("has reward range",     "reward" in spec)
    check("author not placeholder", spec.get("author","") not in ["", "your-hf-username"])
    scores = spec.get("baseline_scores", {})
    check("baseline_scores non-zero", any(v > 0 for v in scores.values()))
except Exception as e:
    check("openenv.yaml parseable", False, str(e))

# 4. Server health
print("\n[4] Server Endpoints (requires server running on localhost:7860)")
try:
    r = requests.get(f"{BASE_URL}/", timeout=5)
    check("GET / returns 200", r.status_code == 200)
except Exception as e:
    check("GET / returns 200", False, str(e))

try:
    r = requests.get(f"{BASE_URL}/state", timeout=5)
    check("GET /state returns 200", r.status_code == 200)
except Exception as e:
    check("GET /state returns 200", False, str(e))

for task_id in [1, 2, 3]:
    try:
        r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=10)
        check(f"POST /reset task_id={task_id}", r.status_code == 200)
        obs = r.json().get("observation", {})
        check(f"  observation has spo2", "spo2" in obs)

        r2 = requests.post(f"{BASE_URL}/step", json={"action": "WAIT_AND_MONITOR"}, timeout=10)
        check(f"POST /step task_id={task_id}", r2.status_code == 200)
        step_data = r2.json()
        reward = step_data.get("reward", None)
        check(f"  reward in [-1,1]", reward is not None and -1.0 <= reward <= 1.0, str(reward))

        r3 = requests.post(f"{BASE_URL}/grade", timeout=10)
        check(f"POST /grade task_id={task_id}", r3.status_code == 200)
        score = r3.json().get("final_score", None)
        check(f"  final_score in [0,1]", score is not None and 0.0 <= score <= 1.0, str(score))
    except Exception as e:
        check(f"task_id={task_id} endpoints", False, str(e))

# 5. inference.py log format
print("\n[5] inference.py Log Format")
try:
    with open("inference.py") as f:
        src = f.read()
    check("[START] log present", "[START]" in src)
    check("[STEP] log present",  "[STEP]"  in src)
    check("[END] log present",   "[END]"   in src)
    check("OpenAI client used",  "OpenAI(" in src)
    check("API_BASE_URL used",   "API_BASE_URL" in src)
    check("MODEL_NAME used",     "MODEL_NAME"   in src)
    check("HF_TOKEN used",       "HF_TOKEN"     in src)
except Exception as e:
    check("inference.py readable", False, str(e))

# Summary
print("\n" + "="*46)
if errors:
    print(f"FAILED — {len(errors)} issue(s) found:")
    for e in errors:
        print(f"  {FAIL} {e}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — ready to submit!")
    sys.exit(0)
