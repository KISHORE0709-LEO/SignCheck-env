import streamlit as st
import requests
import time
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="SignCheck ICU Dashboard", page_icon="🏥", layout="wide")

BASE_URL = "http://localhost:7860"

ACTIONS = [
    "SOUND_WARD_ALARM", "CALL_ATTENDING_DOCTOR", "CALL_ICU_SPECIALIST", 
    "CALL_CODE_BLUE", "CHECK_EQUIPMENT", "CHECK_PATIENT_AIRWAY", 
    "REPOSITION_PATIENT", "START_MANUAL_BAGGING", "ADJUST_OXYGEN_FLOW", 
    "SILENCE_ALARM", "CHECK_IV_LINE", "ADMINISTER_EMERGENCY_MED", 
    "WAIT_AND_MONITOR"
]

if "vitals_history" not in st.session_state:
    st.session_state.vitals_history = []
if "action_history_display" not in st.session_state:
    st.session_state.action_history_display = []
if "obs" not in st.session_state:
    st.session_state.obs = None
if "done" not in st.session_state:
    st.session_state.done = False
if "last_reward" not in st.session_state:
    st.session_state.last_reward = 0.0
if "final_score" not in st.session_state:
    st.session_state.final_score = None
if "passed" not in st.session_state:
    st.session_state.passed = None

def fetch_state():
    """Calls GET /state to poll any external server mutations."""
    try:
        r = requests.get(f"{BASE_URL}/state")
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        pass
    return None

def trigger_grade():
    """Calls POST /grade and locally stores the final assessment output."""
    try:
        r = requests.post(f"{BASE_URL}/grade")
        if r.status_code == 200:
            data = r.json()
            st.session_state.final_score = data["final_score"]
            st.session_state.passed = data["passed"]
        else:
            st.error(f"Failed to grade: {r.text}")
    except Exception as e:
        st.error(f"Failed to connect: {e}")

def trigger_reset(task_id):
    """
    Requests the server to wipe and re-initialize the environment for a specific task.
    Resets the historical dataframe rendering components internally.
    """
    try:
        r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
        if r.status_code == 200:
            data = r.json()
            st.session_state.obs = data["observation"]
            st.session_state.done = False
            st.session_state.last_reward = 0.0
            st.session_state.final_score = None
            st.session_state.passed = None
            vitals = data["observation"].copy()
            vitals["step"] = 0
            st.session_state.vitals_history = [vitals]
            st.session_state.action_history_display = []
    except Exception as e:
        st.error(f"Failed to connect to server: {e}")

def trigger_step(action):
    """
    Submits a requested action payload to the server.
    Takes the resultant StepResult to map new graph coordinates.
    """
    if st.session_state.done:
        st.warning("Simulation is marked as done.")
        return
    try:
        r = requests.post(f"{BASE_URL}/step", json={"action": action})
        if r.status_code == 200:
            data = r.json()
            st.session_state.obs = data["observation"]
            st.session_state.done = data["done"]
            st.session_state.last_reward = data.get("reward", 0.0)
            
            step = st.session_state.obs["step_count"]
            vitals = st.session_state.obs.copy()
            vitals["step"] = step
            st.session_state.vitals_history.append(vitals)
            st.session_state.action_history_display.append(f"Step {step} → {action}")
        else:
            st.error(f"Step Failed: {r.text}")
    except Exception as e:
        st.error(f"Failed to connect to server: {e}")

# ====== Sidebar: Control Panel ======
st.sidebar.title("🏥 SignCheck Controls")
st.sidebar.markdown("---")

task_map = {1: "Task 1 (Power Failure)", 2: "Task 2 (Equipment Malfunction)", 3: "Task 3 (Cardiac Event)"}
selected_task = st.sidebar.selectbox("Select Scenario", [1, 2, 3], format_func=lambda x: task_map[x])

if st.sidebar.button("Start Simulation", use_container_width=True, type="primary"):
    trigger_reset(selected_task)

st.sidebar.markdown("---")
st.sidebar.subheader("Manual Step")
selected_action = st.sidebar.selectbox("Select Action", ACTIONS)
if st.sidebar.button("Next Step", use_container_width=True):
    trigger_step(selected_action)

st.sidebar.metric("Reward", st.session_state.last_reward)

st.sidebar.markdown("---")
if st.sidebar.button("Score / Grade Episode", use_container_width=True):
    trigger_grade()
    
if st.session_state.final_score is not None:
    pass_color = "#00ff41" if st.session_state.passed else "#ff4b4b"
    st.sidebar.markdown(f"**Final Score:** {st.session_state.final_score:.2f}")
    st.sidebar.markdown(f"**Passed:** <span style='color:{pass_color}'>{st.session_state.passed}</span>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Auto Simulation")
auto_run = st.sidebar.checkbox("Run Automatically")

# ====== Auto-Run Logic ======
if auto_run and not st.session_state.done and st.session_state.obs is not None:
    time.sleep(1)
    # Basic Heuristic logic for auto-steps
    obs = st.session_state.obs
    next_action = "WAIT_AND_MONITOR"
    if obs["step_count"] == 0:
        if selected_task == 1: next_action = "SOUND_WARD_ALARM"
        elif selected_task == 2: next_action = "CHECK_PATIENT_AIRWAY"
        elif selected_task == 3: next_action = "CALL_CODE_BLUE"
    elif obs["step_count"] == 1:
        if selected_task == 1: next_action = "CHECK_EQUIPMENT"
        elif selected_task == 2: next_action = "CHECK_EQUIPMENT"
        elif selected_task == 3: next_action = "ADMINISTER_EMERGENCY_MED"
    else:
        if obs.get("spo2", 100) < 90: next_action = "ADJUST_OXYGEN_FLOW"
        else: next_action = "WAIT_AND_MONITOR"
    
    trigger_step(next_action)
    st.rerun()

# ====== Main Dashboard ======
st.title("Live ICU Patient Monitor")

if st.session_state.obs is None:
    st.info("Simulation not initialized. Please click 'Start Simulation' from the Control Panel.")
else:
    obs = st.session_state.obs
    
    st.write("**Simulation Status:**", "COMPLETED" if st.session_state.done else "RUNNING")
    
    critical_any = (
        obs.get("spo2", 100) < 90 or 
        obs.get("heart_rate", 80) > 120 or 
        obs.get("heart_rate", 80) < 50 or 
        obs.get("bp_systolic", 120) < 90 or 
        obs.get("resp_rate", 16) > 25 or 
        obs.get("temperature", 37.0) > 38 or 
        obs.get("consciousness", "Alert") in ["Pain", "Unresponsive"]
    )
    if critical_any:
        st.error("⚠️ CRITICAL CONDITION")
    
    # -- TOP ROW: Vitals Metrics --
    cols = st.columns(6)
    
    def display_metric(col, label, val, unit, is_critical):
        color = "#ff4b4b" if is_critical else "#ffffff"
        col.markdown(f"""
        <div style="background-color:#1e1e1e; padding:15px; border-radius:10px; text-align:center; border: 1px solid {color if is_critical else '#333'}">
            <p style="margin:0;font-size:14px;color:#aaaaaa;font-weight:bold;">{label}</p>
            <h2 style="margin:0;color:{color};">{val} <span style="font-size:14px;color:#888;">{unit}</span></h2>
        </div>
        """, unsafe_allow_html=True)
        
    display_metric(cols[0], "SpO2", f"{obs['spo2']:.1f}", "%", obs['spo2'] < 90)
    display_metric(cols[1], "Heart Rate", f"{obs['heart_rate']}", "bpm", obs['heart_rate'] > 120 or obs['heart_rate'] < 50)
    display_metric(cols[2], "BP Systolic", f"{obs['bp_systolic']}", "mmHg", obs['bp_systolic'] < 90)
    display_metric(cols[3], "Resp Rate", f"{obs['resp_rate']}", "bpm", obs['resp_rate'] > 25)
    display_metric(cols[4], "Temp", f"{obs['temperature']:.1f}", "°C", obs['temperature'] > 38)
    display_metric(cols[5], "AVPU", f"{obs['consciousness']}", "", obs['consciousness'] in ["Pain", "Unresponsive"])

    st.markdown("<br>", unsafe_allow_html=True)

    # -- MIDDLE ROW: Graphs --
    g_cols = st.columns(3)
    
    if len(st.session_state.vitals_history) > 0:
        df = pd.DataFrame(st.session_state.vitals_history)
        
        # SpO2 Plot
        fig_spo2 = go.Figure()
        fig_spo2.add_trace(go.Scatter(x=df['step'], y=df['spo2'], mode='lines+markers', line=dict(color='#00d4ff', width=3)))
        fig_spo2.update_layout(title="SpO2 vs Time (%)", template="plotly_dark", margin=dict(l=20,r=20,t=40,b=20), height=300, paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e")
        g_cols[0].plotly_chart(fig_spo2, use_container_width=True)

        # HR Plot
        fig_hr = go.Figure()
        fig_hr.add_trace(go.Scatter(x=df['step'], y=df['heart_rate'], mode='lines+markers', line=dict(color='#00ff41', width=3)))
        fig_hr.update_layout(title="Heart Rate vs Time (bpm)", template="plotly_dark", margin=dict(l=20,r=20,t=40,b=20), height=300, paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e")
        g_cols[1].plotly_chart(fig_hr, use_container_width=True)

        # BP Plot
        fig_bp = go.Figure()
        fig_bp.add_trace(go.Scatter(x=df['step'], y=df['bp_systolic'], name="Systolic", mode='lines+markers', line=dict(color='#ffea00', width=3)))
        fig_bp.add_trace(go.Scatter(x=df['step'], y=df['bp_diastolic'], name="Diastolic", mode='lines+markers', line=dict(color='#ff9100', width=2, dash='dot')))
        fig_bp.update_layout(title="Blood Pressure vs Time (mmHg)", template="plotly_dark", margin=dict(l=20,r=20,t=40,b=20), height=300, paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e")
        g_cols[2].plotly_chart(fig_bp, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -- BOTTOM ROW: Timeline & Status --
    b_cols = st.columns([1, 1])
    
    with b_cols[0]:
        st.markdown("<h3 style='margin-bottom:0px;'>📋 Status Panel</h3>", unsafe_allow_html=True)
        st.markdown("<hr style='margin-top:10px; margin-bottom:10px;'>", unsafe_allow_html=True)
        
        outcome_color = "#00ff41" if obs["patient_outcome"] in ["stable", "saved"] else ("#ff4b4b" if obs["patient_outcome"] in ["critical", "deceased"] else "#ffea00")
        
        st.markdown(f"**Patient Outcome:** <span style='color:{outcome_color}; font-weight:bold;'>{obs['patient_outcome'].upper()}</span>", unsafe_allow_html=True)
        st.markdown(f"**Step Count:** {obs['step_count']}")
        
        eta = obs.get('doctor_eta')
        st.markdown(f"**Doctor ETA:** {eta if eta is not None else 'Not Alerted'} steps")
        st.markdown(f"**Equipment:** {obs['equipment_status']}")
        st.markdown(f"**Power Status:** {obs['power_status']}")
        
        st.info(f"**Last Action Feedback:** {obs['last_action_feedback']}")
        st.warning(f"**Clinical Notes:** {obs['clinical_notes']}")
        
    with b_cols[1]:
        st.markdown("<h3 style='margin-bottom:0px;'>⏱️ Action Timeline</h3>", unsafe_allow_html=True)
        st.markdown("<hr style='margin-top:10px; margin-bottom:10px;'>", unsafe_allow_html=True)
        
        timeline_container = st.container(height=300)
        if len(st.session_state.action_history_display) == 0:
            timeline_container.write("No actions taken yet.")
        else:
            for act in reversed(st.session_state.action_history_display):
                timeline_container.markdown(f"> **{act}**")
