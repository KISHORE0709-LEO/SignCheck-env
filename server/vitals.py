import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

@dataclass
class VitalSigns:
    spo2: float          # normal 95-100
    heart_rate: int      # normal 60-100
    bp_systolic: int     # normal 110-130
    bp_diastolic: int    # normal 70-85
    resp_rate: int       # normal 12-20
    temperature: float   # normal 36.5-37.5
    consciousness: str   # AVPU: Alert/Voice/Pain/Unresponsive

@dataclass
class VitalThresholds:
    spo2_low: float = 88.0
    heart_rate_high: int = 140
    heart_rate_low: int = 45
    bp_systolic_low: int = 90
    bp_systolic_high: int = 180
    bp_diastolic_low: int = 60
    bp_diastolic_high: int = 110
    resp_rate_low: int = 8
    resp_rate_high: int = 30
    temp_low: float = 35.0
    temp_high: float = 39.0

def apply_drift(vitals: VitalSigns, drift_rates: Dict[str, float], noise: bool = True) -> Tuple[VitalSigns, Dict[str, Any]]:
    # Apply standard drift + noise
    def _add_noise(base: float, amount: float) -> float:
        return base + amount + (random.gauss(0, 0.5) if noise else 0.0)
    
    # Store previous for logic checks
    prev_spo2 = vitals.spo2
    prev_hr = vitals.heart_rate

    # Calculate new values
    vitals.spo2 = _add_noise(vitals.spo2, drift_rates.get('spo2', 0.0))
    vitals.heart_rate = int(_add_noise(vitals.heart_rate, drift_rates.get('heart_rate', 0.0)))
    vitals.bp_systolic = int(_add_noise(vitals.bp_systolic, drift_rates.get('bp_systolic', 0.0)))
    vitals.bp_diastolic = int(_add_noise(vitals.bp_diastolic, drift_rates.get('bp_diastolic', 0.0)))
    vitals.resp_rate = int(_add_noise(vitals.resp_rate, drift_rates.get('resp_rate', 0.0)))
    vitals.temperature = _add_noise(vitals.temperature, drift_rates.get('temperature', 0.0))
    
    effects = {}
    
    # Cascading Consequences:
    # If spo2 dropped below 88 previously, consciousness degrades one level this call
    if prev_spo2 < 88.0:
        if vitals.consciousness == "Alert":
            vitals.consciousness = "Voice"
            effects["consciousness_degraded"] = "SpO2 < 88%: Consciousness degraded to Voice."
        elif vitals.consciousness == "Voice":
            vitals.consciousness = "Pain"
            effects["consciousness_degraded"] = "SpO2 < 88%: Consciousness degraded to Pain."
        elif vitals.consciousness == "Pain":
            vitals.consciousness = "Unresponsive"
            effects["consciousness_degraded"] = "SpO2 < 88%: Consciousness degraded to Unresponsive."

    # If heart_rate exceeds 150 previously, bp_systolic drops by 3 this call
    if prev_hr > 150:
        vitals.bp_systolic -= 3
        effects["cascading_bp_drop"] = "Tachycardia (HR > 150) caused subsequent drop in BP Systolic by 3."
        
    return vitals, effects

def compute_stability_score(vitals: VitalSigns) -> float:
    normal_count = 0
    total_metrics = 6
    
    if 95.0 <= vitals.spo2 <= 100.0: normal_count += 1
    if 60 <= vitals.heart_rate <= 100: normal_count += 1
    if 110 <= vitals.bp_systolic <= 130: normal_count += 1
    if 70 <= vitals.bp_diastolic <= 85: normal_count += 1
    if 12 <= vitals.resp_rate <= 20: normal_count += 1
    if 36.5 <= vitals.temperature <= 37.5: normal_count += 1
    
    return float(normal_count) / float(total_metrics)

def check_critical(vitals: VitalSigns) -> List[str]:
    criticals = []
    t = VitalThresholds()
    
    if vitals.spo2 < t.spo2_low:
        criticals.append("spo2")
    if vitals.heart_rate < t.heart_rate_low or vitals.heart_rate > t.heart_rate_high:
        criticals.append("heart_rate")
    if vitals.bp_systolic < t.bp_systolic_low or vitals.bp_systolic > t.bp_systolic_high:
        criticals.append("bp_systolic")
    if vitals.bp_diastolic < t.bp_diastolic_low or vitals.bp_diastolic > t.bp_diastolic_high:
        criticals.append("bp_diastolic")
    if vitals.resp_rate < t.resp_rate_low or vitals.resp_rate > t.resp_rate_high:
        criticals.append("resp_rate")
    if vitals.temperature < t.temp_low or vitals.temperature > t.temp_high:
        criticals.append("temperature")
    if vitals.consciousness == "Unresponsive":
        criticals.append("consciousness")
        
    return criticals
