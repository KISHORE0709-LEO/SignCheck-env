"""
playground.py — SignCheck-env heuristic debug runner.

Usage:
    python playground.py --task 1    # run Task 1 (default)
    python playground.py --task 2
    python playground.py --task 3
"""
import argparse
import requests

BASE_URL = "http://localhost:7860"


def heuristic(obs: dict, step: int) -> str:
    if obs["spo2"] < 92:
        return "ADJUST_OXYGEN_FLOW"
    if obs["heart_rate"] > 120:
        return "CHECK_PATIENT_AIRWAY"
    if step > 5:
        return "CALL_ATTENDING_DOCTOR"
    return "CHECK_EQUIPMENT"


def run(task_id: int):
    print(f"\nResetting environment — task_id={task_id}")
    r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    data = r.json()
    obs = data["observation"]
    print(f"Task: {data['task_description']}\n")
    print(f"{'Step':<6} {'Action':<28} {'SpO2':>6} {'HR':>5} {'Reward':>8} {'Done'}")
    print("-" * 65)

    step = 0
    done = False
    while not done:
        step += 1
        action = heuristic(obs, step)
        r = requests.post(f"{BASE_URL}/step", json={"action": action}, timeout=30)
        r.raise_for_status()
        sd = r.json()
        obs    = sd["observation"]
        reward = sd["reward"]
        done   = sd["done"]
        print(
            f"{step:<6} {action:<28} {obs['spo2']:>5.1f}% {obs['heart_rate']:>4}bpm "
            f"{reward:>8.3f}  {'✓' if done else ''}"
        )

    print("\nEpisode finished. Grading...")
    r = requests.post(f"{BASE_URL}/grade", timeout=30)
    r.raise_for_status()
    g = r.json()
    print(f"\n{'='*40}")
    print(f"  Final Score  : {g['final_score']:.3f}")
    print(f"  Survival     : {g['survival_score']:.3f}")
    print(f"  Stability    : {g['stability_score']:.3f}")
    print(f"  Escalation   : {g['escalation_score']:.3f}")
    print(f"  Efficiency   : {g['efficiency_score']:.3f}")
    print(f"  Passed       : {'✅' if g['passed'] else '❌'}")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignCheck-env heuristic playground")
    parser.add_argument("--task", type=int, default=1, choices=[1, 2, 3],
                        help="Task ID to run (1=easy, 2=medium, 3=hard)")
    args = parser.parse_args()
    run(args.task)
