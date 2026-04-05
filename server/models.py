from enum import Enum
from pydantic import BaseModel
from typing import Dict, Optional, Any

class Action(str, Enum):
    SOUND_WARD_ALARM = "SOUND_WARD_ALARM"
    CALL_ATTENDING_DOCTOR = "CALL_ATTENDING_DOCTOR"
    CALL_ICU_SPECIALIST = "CALL_ICU_SPECIALIST"
    CALL_CODE_BLUE = "CALL_CODE_BLUE"
    CHECK_EQUIPMENT = "CHECK_EQUIPMENT"
    CHECK_PATIENT_AIRWAY = "CHECK_PATIENT_AIRWAY"
    REPOSITION_PATIENT = "REPOSITION_PATIENT"
    START_MANUAL_BAGGING = "START_MANUAL_BAGGING"
    ADJUST_OXYGEN_FLOW = "ADJUST_OXYGEN_FLOW"
    SILENCE_ALARM = "SILENCE_ALARM"
    CHECK_IV_LINE = "CHECK_IV_LINE"
    ADMINISTER_EMERGENCY_MED = "ADMINISTER_EMERGENCY_MED"
    WAIT_AND_MONITOR = "WAIT_AND_MONITOR"

class PatientOutcome(str, Enum):
    STABLE = "stable"
    DETERIORATING = "deteriorating"  
    CRITICAL = "critical"
    DECEASED = "deceased"
    SAVED = "saved"

class Observation(BaseModel):
    spo2: float
    heart_rate: int
    bp_systolic: int
    bp_diastolic: int
    resp_rate: int
    temperature: float
    consciousness: str  # AVPU scale
    equipment_status: dict
    power_status: str
    time_elapsed: int
    time_since_last_vitals_check: int
    doctor_eta: Optional[int] = None
    clinical_notes: str
    last_action_feedback: str
    step_count: int
    patient_outcome: PatientOutcome = PatientOutcome.STABLE

class Reward(BaseModel):
    reward: float
    message: str

# info dict will always contain:
# {"survival": float, "stability": float, "escalation": float, "efficiency": float, "outcome": str}
class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict

class ResetResult(BaseModel):
    observation: Observation
    task_id: int
    task_description: str
