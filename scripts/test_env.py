"""
test_env.py — Smoke-test all SignCheck-env endpoints.

Exits with code 0 on success, 1 on any failure.

Usage:
    python test_env.py
"""
import sys
import requests

BASE_URL = "http://localhost:7860"
PASS = "✅"
FAIL = "❌"


def check(name: str, fn):
    try:
        result = fn()
        print(f"  {PASS}  {name}")
        return result
    except Exception as e:
        print(f"  {FAIL}  {name}  →  {e}")
        sys.exit(1)


def main():
    print("\nSignCheck-env smoke test")
    print("=" * 40)

    # 1. /health
    check("/health", lambda: (
        requests.get(f"{BASE_URL}/health", timeout=10).raise_for_status()
    ))

    # 2. /tasks
    tasks = check("/tasks", lambda: (
        requests.get(f"{BASE_URL}/tasks", timeout=10).json()
    ))
    assert isinstance(tasks, list) and len(tasks) == 3, "Expected 3 tasks"

    # 3. /reset
    reset_data = check("/reset", lambda: (
        requests.post(f"{BASE_URL}/reset", json={"task_id": 1}, timeout=10).json()
    ))
    assert "observation" in reset_data, "Missing 'observation' in /reset response"

    # 4. /step
    step_data = check("/step", lambda: (
        requests.post(f"{BASE_URL}/step", json={"action": "CHECK_EQUIPMENT"}, timeout=10).json()
    ))
    assert "reward" in step_data, "Missing 'reward' in /step response"
    assert "done" in step_data,   "Missing 'done' in /step response"

    # 5. /state
    check("/state", lambda: (
        requests.get(f"{BASE_URL}/state", timeout=10).raise_for_status()
    ))

    # 6. /grade
    grade_data = check("/grade", lambda: (
        requests.post(f"{BASE_URL}/grade", timeout=10).json()
    ))
    assert "final_score" in grade_data, "Missing 'final_score' in /grade response"

    print("=" * 40)
    print("All endpoints OK — environment is healthy.\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
