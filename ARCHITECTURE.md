# SignCheck-Env Architecture

SignCheck-Env follows a centralized client-server architecture mapping clinical events into a standard Reinforcement Learning (MDP) loop. The simulation runs entirely decoupled from the client agents.

## Core Components

- **Client Agent** (`agent/inference.py`): Reaches out to the server, parses `clinical_notes` and vitals, and sends an `Action`.
- **FastAPI Server** (`server/main.py`): Holds the master state and handles incoming agent configurations.
- **Simulation Engine** (`server/env.py` & `server/vitals.py`): Performs deterministic vital degradation, adds localized noise, applies interventions, and cascades medical consequences if the patient is ignored.

## Environment Flow

### What happens when `/reset` is called?
1. The server receives a `task_id` (1, 2, or 3).
2. It loads the respective physiological baselines from `server/scenarios.py`.
3. It initializes the `VitalSigns` dataclass, zeros out the clock (`step_count`), and generates an initial observation containing real-time vitals and text-based clinical notes.

### What happens during `/step(action)`?
1. The engine checks if the requested action maps to any immediate physiological interventions (e.g., `START_MANUAL_BAGGING` increases `SpO2`).
2. If an escalation action is taken (e.g., `CALL_CODE_BLUE`), the `doctor_eta` timer is triggered. If the wrong escalation is chosen, a massive penalty and doubled wait-time is applied.
3. The engine applies the scenario's standard vital drift (worsening conditions per step) to all unstabilized variables.
4. The engine checks cascading rules: "Did SpO2 drop below 88? If so, lower consciousness level."
5. A step reward is calculated, and the new state is returned.

### How is the Reward calculated?
The reward (`-1.0` to `1.0`) is continuously calculated per step using a composite formula:
- **Stability Delta**: Did the combined health of the 6 core vitals improve or worsen since last turn?
- **Action Bonus**: Direct bonuses/penalties for applying specific localized treatments correctly or incorrectly.
- **Escalation**: Bonuses for choosing the correct level of hospital escalation at the perfect time.
- **Delay Penalty**: Subtractions if the agent idles on `WAIT_AND_MONITOR` during a crash.

### How does Grading work (`/grade`)?
When the episode limits out (either patient saved, died, or time ran out), the grader evaluates the full historical trajectory:
1. **Survival**: Saved = `1.0`, Stable = `0.5`, Deceased = `0.0`.
2. **Stability**: Mean percentage of vitals held in the normal range over the *entire* episode.
3. **Escalation Quality**: Max points if the exact correct physician was alerted early. Halved if alerted late. Zeroed if ignored.
4. **Efficiency**: Remaining spare steps divided by total allowed steps.
