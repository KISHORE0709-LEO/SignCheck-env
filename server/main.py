from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import copy

from .env import SignCheckEnv
from .models import Action, ResetResult, StepResult
from .vitals import VitalSigns
from .scenarios import get_all_scenarios
from .grader import grade_episode, GradeResult

app = FastAPI(
    title="SignCheck-Env: ICU Emergency Response RL Environment",
    description="An OpenEnv reinforcement learning environment where an AI agent stabilizes a patient during hospital emergencies until a doctor arrives."
)

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

    model_config = {"json_schema_extra": {"example": {"action": "CHECK_EQUIPMENT"}}}

@app.on_event("startup")
def startup_event():
    """Initializes the environment internally on server startup."""
    global vital_log
    env.reset(1)
    vital_log = [copy.deepcopy(env.vitals)]

@app.get("/health", summary="Health Check")
def health():
    """Returns the basic health status of the API server."""
    return {"status": "ok"}

@app.get("/", summary="Root Endpoint")
def root():
    """Returns a simple JSON message containing the project name, a short description, and a link to /docs."""
    return {
        "project": "SignCheck-Env",
        "description": "ICU Emergency Response RL Environment",
        "message": "Welcome to SignCheck-Env! Please visit /docs for interactive documentation.",
        "docs_url": "/docs",
        "endpoints": ["/reset", "/step", "/state", "/grade", "/tasks"]
    }


@app.get("/tasks", summary="List Tasks")
def get_tasks():
    """Returns a list of all available ICU emergency scenarios and their details."""
    scenarios = get_all_scenarios()
    return [{
        "task_id": s["task_id"],
        "name": s["name"],
        "difficulty": s["difficulty"],
        "description": s["description"],
        "max_steps": s["max_steps"]
    } for s in scenarios]

@app.post("/reset", response_model=ResetResult, summary="Start New Episode")
def reset_env(params: Optional[ResetParams] = None):
    """
    Starts a new episode for the specified Task ID.
    Returns the initial patient observation and the scenario context.
    """
    global vital_log
    task_id = params.task_id if params else 1
    try:
        res = env.reset(task_id)
        vital_log = [copy.deepcopy(env.vitals)]
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step", response_model=StepResult, summary="Apply Action to Environment")
def step_env(params: StepParams):
    """
    Applies an action to the environment, updates the patient vitals using the simulator, 
    and steps the simulation time forward. 
    Returns the new observation, the generated reward, and the done flag determining if the episode is finished.
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

@app.get("/state", summary="Get Current State")
def state():
    """Retrieves the raw internal dictionary state safely without advancing the simulation clock."""
    return env.state()

@app.post("/grade", response_model=GradeResult, summary="Grade Episode")
def grade():
    """
    Evaluates the agent's complete performance for the current episode 
    and returns deterministic grading metrics between 0 and 1.
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
