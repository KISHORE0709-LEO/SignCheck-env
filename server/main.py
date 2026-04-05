from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import copy

from .env import SignCheckEnv
from .models import Action, ResetResult, StepResult
from .vitals import VitalSigns
from .scenarios import get_all_scenarios
from .grader import grade_episode, GradeResult

app = FastAPI(title="SignCheck-env")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = SignCheckEnv()
vital_log: List[VitalSigns] = []

class ResetParams(BaseModel):
    task_id: int = 1

class StepParams(BaseModel):
    action: str

@app.on_event("startup")
def startup_event():
    """Initializes the environment internally on server startup."""
    global vital_log
    env.reset(1)
    vital_log = [copy.deepcopy(env.vitals)]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def get_tasks():
    scenarios = get_all_scenarios()
    return [{
        "task_id": s["task_id"],
        "name": s["name"],
        "difficulty": s["difficulty"],
        "description": s["description"],
        "max_steps": s["max_steps"]
    } for s in scenarios]

@app.post("/reset", response_model=ResetResult)
def reset_env(params: ResetParams):
    """
    Resets the environment to a specific Task ID.
    Returns: initial Observation and task setup data.
    """
    global vital_log
    try:
        res = env.reset(params.task_id)
        vital_log = [copy.deepcopy(env.vitals)]
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step", response_model=StepResult)
def step_env(params: StepParams):
    """
    Applies the specified Action to the state instance.
    Calculates drift, triggers consequences, and scores the current turn.
    Returns: new Observation, Reward, and Terminal Done flag.
    """
    try:
        action_enum = Action(params.action)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid action: {params.action}")
        
    try:
        res = env.step(action_enum)
        vital_log.append(copy.deepcopy(env.vitals))
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
def state():
    """Returns the raw internal dictionary state (vitals, history, metadata)."""
    return env.state()

@app.post("/grade", response_model=GradeResult)
def grade():
    """
    Grades the entire previous episode based on final outcome, history, 
    and task stability thresholds. 
    Returns: Component metrics and final 0-1 score block.
    """
    if env.scenario is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
        
    res = grade_episode(
        task_id=env.task_id,
        action_history=env.action_history,
        vital_log=vital_log,
        outcome=env.patient_outcome,
        step_count=env.step_count,
        max_steps=env.scenario.get("max_steps", 30)
    )
    return res
