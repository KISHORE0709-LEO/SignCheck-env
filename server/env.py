import random
import copy
from typing import Dict, Any, Tuple

from .models import (Action, Observation, StepResult, ResetResult, 
                     PatientOutcome, Reward)
from .vitals import VitalSigns, apply_drift, compute_stability_score, check_critical
from .scenarios import get_scenario, get_intervention_effect

class SignCheckEnv:
    def __init__(self):
        self.task_id = 1
        self.scenario = None
        self.vitals = None
        
        self.step_count = 0
        self.time_elapsed = 0
        self.time_since_last_vitals_check = 0
        self.doctor_called = False
        self.doctor_eta = None
        self.drift_frozen = False
        self.false_alarm_is_sensor_error = False
        self.action_history = []
        
        self.equipment_status = {}
        self.power_status = ""
        self.clinical_notes = ""
        self.last_action_feedback = ""
        self.patient_outcome = PatientOutcome.STABLE

    def reset(self, task_id: int) -> ResetResult:
        """
        Loads the provided scenario baseline parameters into the physics engine
        and resets all runtime variables (counters, timers, flags).
        
        Outputs the standard starting observation structure.
        """
        self.task_id = task_id
        self.scenario = get_scenario(task_id)
        
        self.vitals = VitalSigns(**self.scenario["initial_vitals"])
        
        self.step_count = 0
        self.time_elapsed = 0
        self.time_since_last_vitals_check = 0
        self.doctor_called = False
        self.doctor_eta = None
        self.drift_frozen = False
        
        chance = self.scenario.get("false_alarm_chance", 0.0)
        self.false_alarm_is_sensor_error = (random.random() < chance)
        
        self.action_history = []
        self.equipment_status = copy.deepcopy(self.scenario.get("equipment_status_initial", {}))
        self.power_status = self.scenario.get("power_status_initial", "normal")
        self.clinical_notes = self.scenario.get("clinical_notes_initial", "")
        self.last_action_feedback = "Environment initialized."
        self.patient_outcome = PatientOutcome.STABLE

        return ResetResult(
            observation=self._build_observation(),
            task_id=self.task_id,
            task_description=self.scenario["description"]
        )

    def _build_observation(self) -> Observation:
        return Observation(
            spo2=round(self.vitals.spo2, 1),
            heart_rate=int(self.vitals.heart_rate),
            bp_systolic=int(self.vitals.bp_systolic),
            bp_diastolic=int(self.vitals.bp_diastolic),
            resp_rate=int(self.vitals.resp_rate),
            temperature=round(self.vitals.temperature, 1),
            consciousness=self.vitals.consciousness,
            equipment_status=copy.deepcopy(self.equipment_status),
            power_status=self.power_status,
            time_elapsed=self.time_elapsed,
            time_since_last_vitals_check=self.time_since_last_vitals_check,
            doctor_eta=self.doctor_eta,
            clinical_notes=self.clinical_notes,
            last_action_feedback=self.last_action_feedback,
            step_count=self.step_count,
            patient_outcome=self.patient_outcome
        )

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step_count": self.step_count,
            "vitals": self.vitals.__dict__,
            "history": [a.value for a in self.action_history],
            "outcome": self.patient_outcome.value
        }

    def _get_action_feedback(self, action: Action) -> str:
        feedback = f"Performed {action.value}."
        
        if self.task_id == 2 and self.false_alarm_is_sensor_error:
            if action == Action.CHECK_EQUIPMENT:
                feedback += " Discovered the E2 alarm is a sensor false alarm, physical airway is clear."
                
        if action == Action.WAIT_AND_MONITOR:
            feedback = "Waited and monitored patient."
        return feedback

    def _update_clinical_notes(self, cascading_effects: Dict[str, str]) -> str:
        notes = self.scenario.get("clinical_notes_initial", "")
        if self.doctor_called:
            notes += f" Doctor expected in {self.doctor_eta} min."
        if cascading_effects:
            notes += " " + " ".join(cascading_effects.values())
            
        criticals = check_critical(self.vitals)
        if len(criticals) > 0:
            notes += f" Critical values detected in: {', '.join(criticals)}."
            
        return notes.strip()

    def _compute_escalation_bonus(self, action: Action) -> float:
        expected_seq = self.scenario.get("correct_action_sequence", [])
        if self.step_count < len(expected_seq):
            if action == expected_seq[self.step_count]:
                return 0.2
            elif action in expected_seq:
                return 0.1
        return 0.0

    def _compute_reward(
        self, action: Action, 
        prev_vitals: VitalSigns, 
        cascading_effects: Dict[str, str],
        action_bonus: float
    ) -> float:
        prev_stability = compute_stability_score(prev_vitals)
        curr_stability = compute_stability_score(self.vitals)
        stability_delta = curr_stability - prev_stability
        
        escalation_bonus = self._compute_escalation_bonus(action)
        delay_penalty = 1.0 if action == Action.WAIT_AND_MONITOR else 0.0
        
        total = (stability_delta * 0.3) + (action_bonus * 0.2) - (delay_penalty * 0.05) + escalation_bonus
        return round(max(-1.0, min(1.0, total)), 3)

    def _check_terminal(self) -> Tuple[bool, PatientOutcome]:
        criticals = check_critical(self.vitals)
        
        if len(criticals) >= 2:
            if self.vitals.consciousness == "Unresponsive":
                return True, PatientOutcome.DECEASED
            return True, PatientOutcome.CRITICAL
            
        if self.step_count >= self.scenario["max_steps"]:
            return True, PatientOutcome.DETERIORATING
            
        if self.doctor_called and self.doctor_eta is not None and self.doctor_eta <= 0:
            if compute_stability_score(self.vitals) >= 0.5 and len(criticals) == 0:
                return True, PatientOutcome.SAVED
            else:
                return True, PatientOutcome.DECEASED
                
        return False, PatientOutcome.STABLE

    def step(self, action: Action) -> StepResult:
        """
        The core physics engine heartbeat.
        Takes a verified Action enum integer mapping to apply:
        1. Base drift mechanics + stochastic vital deviations
        2. Escalation and medical response effects
        3. Cascading consequences based on previous step vitals (MDP)
        
        Calculates and returns the isolated Reward array.
        """
        if self.scenario is None:
            raise ValueError("Environment must be reset before stepping.")
            
        self.step_count += 1
        self.time_elapsed += 1
        self.time_since_last_vitals_check += 1
        self.action_history.append(action)
        
        if self.doctor_called and self.doctor_eta is not None:
            self.doctor_eta = max(0, self.doctor_eta - 1)
            
        prev_vitals = copy.deepcopy(self.vitals)
        action_bonus = 0.0

        # Apply continuous drift FIRST
        cascading_effects = {}
        if not self.drift_frozen:
            drifts = self.scenario.get("drift_rates", {})
            self.vitals, cascading_effects = apply_drift(self.vitals, drifts, noise=True)

        # Support basic interventions
        intervention = get_intervention_effect(self.scenario, action)
        if "spo2" in intervention:
            self.vitals.spo2 += intervention["spo2"]
        if "heart_rate" in intervention:
            self.vitals.heart_rate += int(intervention["heart_rate"])
            
        if intervention.get("drift_freeze"):
            self.drift_frozen = True
            
        if "stabilize_equipment" in intervention:
            if "ventilator" in self.equipment_status:
                self.equipment_status["ventilator"] = "normal"
                
        if "alarms_active" in intervention:
            self.equipment_status["monitor"] = "alarms_silenced"

        if "doctor_alerted" in intervention and not self.doctor_called:
            self.doctor_called = True
            self.doctor_eta = self.scenario.get("doctor_eta_initial", 5)
            if intervention.get("_trap_code") == "wrong_escalation":
                self.doctor_eta *= 2 # delayed arrival as punishment
                action_bonus = -2.5
        
        # Support special med logic from hard scenario
        spec_med = intervention.get("_special_med_effect", {})
        if spec_med:
            if random.random() < spec_med.get("hr_stabilize_chance", 0.0):
                self.vitals.heart_rate += spec_med.get("hr_stabilize_val", 0)
                action_bonus += 2.5
            elif random.random() < spec_med.get("bp_crash_chance", 0.0):
                self.vitals.bp_systolic += spec_med.get("bp_crash_val", 0)
                action_bonus -= 2.5
                
        # Support false alarm Task 2 logic
        if self.task_id == 2:
            if self.false_alarm_is_sensor_error:
                if action == Action.CHECK_EQUIPMENT:
                    action_bonus += 0.75
                elif action == Action.START_MANUAL_BAGGING and Action.CHECK_EQUIPMENT not in self.action_history:
                    action_bonus -= 0.5
                    
        # Reset observation timers
        if action in [Action.CHECK_EQUIPMENT, Action.CHECK_PATIENT_AIRWAY, Action.WAIT_AND_MONITOR]:
            self.time_since_last_vitals_check = 0
            
        # Compile text strings
        self.last_action_feedback = self._get_action_feedback(action)
        self.clinical_notes = self._update_clinical_notes(cascading_effects)
        
        # Terminal checks and Rewards
        reward_val = self._compute_reward(action, prev_vitals, cascading_effects, action_bonus)
        done, outcome = self._check_terminal()
        self.patient_outcome = outcome
        
        info = {
            "survival": 1.0 if outcome == PatientOutcome.SAVED else 0.0,
            "stability": compute_stability_score(self.vitals),
            "escalation": self._compute_escalation_bonus(action),
            "efficiency": max(0.0, 1.0 - (self.step_count / self.scenario.get("max_steps", 30))),
            "outcome": outcome.value,
        }
        
        return StepResult(
            observation=self._build_observation(),
            reward=reward_val,
            done=done,
            info=info
        )
