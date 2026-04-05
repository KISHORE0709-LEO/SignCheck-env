from typing import Dict, Any, List
from .models import Action

def get_scenario(task_id: int) -> Dict[str, Any]:
    if task_id == 1:
        return {
            "task_id": 1,
            "name": "Power Failure",
            "difficulty": "easy",
            "description": "Hospital power failure causing equipment to run on backup battery. Slow vital degradation expected.",
            "initial_vitals": {
                "spo2": 96.0,
                "heart_rate": 80,
                "bp_systolic": 115,
                "bp_diastolic": 75,
                "resp_rate": 16,
                "temperature": 36.8,
                "consciousness": "Alert"
            },
            "drift_rates": {
                "spo2": -0.3,
                "heart_rate": 1.0,
                "bp_systolic": -0.5,
                "bp_diastolic": -0.5,
                "resp_rate": 0.0,
                "temperature": 0.0
            },
            "action_interventions": {
                Action.ADJUST_OXYGEN_FLOW: {"spo2": 1.0},
                Action.CHECK_EQUIPMENT: {"stabilize_equipment": True},
                Action.SILENCE_ALARM: {"alarms_active": False},
                Action.CALL_ATTENDING_DOCTOR: {"doctor_alerted": True}
            },
            "equipment_status_initial": {
                "monitor": "offline", 
                "iv_pump": "beeping", 
                "ventilator": "on_battery"
            },
            "power_status_initial": "brownout",
            "doctor_eta_initial": 12,
            "correct_action_sequence": [
                Action.SOUND_WARD_ALARM, 
                Action.CHECK_EQUIPMENT, 
                Action.ADJUST_OXYGEN_FLOW, 
                Action.CALL_ATTENDING_DOCTOR
            ],
            "clinical_notes_initial": "Nurse notes: Ward experienced sudden brownout. Patient stable, but ventilator is on battery and monitor offline.",
            "max_steps": 30,
            "false_alarm_chance": 0.0
        }
    elif task_id == 2:
        return {
            "task_id": 2,
            "name": "Ventilator Malfunction",
            "difficulty": "medium",
            "description": "Ventilator alarm sounding. Agent must distinguish equipment obstruction vs. sensor error before treating.",
            "initial_vitals": {
                "spo2": 92.0,
                "heart_rate": 105,
                "bp_systolic": 125,
                "bp_diastolic": 82,
                "resp_rate": 22,
                "temperature": 37.1,
                "consciousness": "Alert"
            },
            "drift_rates": {
                "spo2": -0.8,
                "heart_rate": 2.5,
                "bp_systolic": 1.5,
                "bp_diastolic": 1.0,
                "resp_rate": 1.5,
                "temperature": 0.0
            },
            "action_interventions": {
                Action.START_MANUAL_BAGGING: {"spo2": 2.0, "heart_rate": -1.0},
                Action.CHECK_PATIENT_AIRWAY: {"airway_checked": True},
                Action.CHECK_EQUIPMENT: {"equipment_checked": True},
                Action.CALL_ICU_SPECIALIST: {"doctor_alerted": True}
            },
            "equipment_status_initial": {
                "monitor": "normal", 
                "ventilator": "alarm E2 occlusion error"
            },
            "power_status_initial": "normal",
            "doctor_eta_initial": 10,
            "correct_action_sequence": [
                Action.CHECK_PATIENT_AIRWAY,
                Action.CHECK_EQUIPMENT,
                Action.START_MANUAL_BAGGING,
                Action.CALL_ICU_SPECIALIST
            ],
            "clinical_notes_initial": "Nurse notes: Ventilator alarming E2 occlusion. Patient appears agitated and diaphoretic. Need to verify airway vs false alarm before intervening.",
            "max_steps": 30,
            "false_alarm_chance": 0.3
        }
    elif task_id == 3:
        return {
            "task_id": 3,
            "name": "Sudden Cardiac Event",
            "difficulty": "hard",
            "description": "Patient enters sudden erratic cardiac rhythm. Trap: Call attending or Code Blue?",
            "initial_vitals": {
                "spo2": 90.0,
                "heart_rate": 140, 
                "bp_systolic": 95,
                "bp_diastolic": 60,
                "resp_rate": 28,
                "temperature": 36.5,
                "consciousness": "Voice"
            },
            "drift_rates": {
                "spo2": -1.5,
                "heart_rate": 15.0, # Added high directional drift; noise will be overlaid in env
                "bp_systolic": -4.0,
                "bp_diastolic": -2.0,
                "resp_rate": 3.0,
                "temperature": 0.0
            },
            "action_interventions": {
                Action.CALL_CODE_BLUE: {"drift_freeze": True},
                Action.ADMINISTER_EMERGENCY_MED: {
                    "_special_med_effect": {
                        "hr_stabilize_chance": 0.60,
                        "hr_stabilize_val": -40,
                        "bp_crash_chance": 0.40,
                        "bp_crash_val": -10
                    }
                },
                Action.START_MANUAL_BAGGING: {"spo2": 2.0},
                Action.CALL_ATTENDING_DOCTOR: {"_trap_code": "wrong_escalation"}
            },
            "equipment_status_initial": {
                "monitor": "showing VFib pattern", 
                "ventilator": "normal",
                "iv_pump": "normal"
            },
            "power_status_initial": "normal",
            "doctor_eta_initial": 8,
            "correct_action_sequence": [
                Action.CALL_CODE_BLUE,
                Action.ADMINISTER_EMERGENCY_MED,
                Action.START_MANUAL_BAGGING
            ],
            "clinical_notes_initial": "Nurse notes: Patient suddenly clutching chest and unresponsive, monitor showing erratic VFib pattern. Immediate escalation required.",
            "max_steps": 25,
            "false_alarm_chance": 0.0
        }
    else:
        raise ValueError(f"Task ID {task_id} not found.")

def get_all_scenarios() -> List[Dict[str, Any]]:
    return [get_scenario(1), get_scenario(2), get_scenario(3)]
