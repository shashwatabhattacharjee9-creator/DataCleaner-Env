"""
FastAPI Server for DataCleaner-Env - A stateful data cleaning environment.
Built for Meta PyTorch Hackathon.
"""

import json
import copy
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from models import (
    Observation, Action, Reward,
    ResetRequest, ResetResponse,
    StepRequest, StepResponse,
    StateRequest, StateResponse
)


app = FastAPI(
    title="DataCleaner-Env API",
    description="Stateful data cleaning environment for RL agents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global state management: maps task_id to session state
SESSIONS: Dict[str, Dict[str, Any]] = {}

# Available actions for the agent
AVAILABLE_ACTIONS = [
    "DROP_NULLS",
    "FILL_NULLS",
    "REPLACE_VALUE",
    "DROP_DUPLICATES",
    "DROP_INVALID_ROWS"
]


def get_initial_dataset(task_id: str) -> List[Dict[str, Any]]:
    """
    Returns the initial dataset for each task.
    """
    datasets = {
        "triage-clean-easy": [
            {"patient_id": 1, "age": 25, "diagnosis": "flu", "admission_date": "2024-01-01"},
            {"patient_id": 2, "age": None, "diagnosis": "cold", "admission_date": "2024-01-02"},
            {"patient_id": 3, "age": 45, "diagnosis": "pneumonia", "admission_date": "2024-01-03"},
            {"patient_id": 1, "age": 25, "diagnosis": "flu", "admission_date": "2024-01-01"},  # Duplicate
            {"patient_id": 4, "age": None, "diagnosis": "bronchitis", "admission_date": "2024-01-04"},
            {"patient_id": 5, "age": 60, "diagnosis": "asthma", "admission_date": "2024-01-05"},
            {"patient_id": 3, "age": 45, "diagnosis": "pneumonia", "admission_date": "2024-01-03"},  # Duplicate
        ],
        "triage-clean-medium": [
            {"patient_id": 1, "age": "twenty-five", "diagnosis": "flu", "severity": "low"},
            {"patient_id": 2, "age": 30, "diagnosis": "cold", "severity": "low"},
            {"patient_id": 3, "age": "forty-five", "diagnosis": "pneumonia", "severity": "high"},
            {"patient_id": 4, "age": 55, "diagnosis": "bronchitis", "severity": "medium"},
            {"patient_id": 5, "age": "sixty", "diagnosis": "asthma", "severity": "medium"},
            {"patient_id": 6, "age": 70, "diagnosis": "copd", "severity": "high"},
        ],
        "triage-clean-hard": [
            {"patient_id": 1, "admission_date": "2024-01-05", "discharge_date": "2024-01-10", "diagnosis": "flu"},
            {"patient_id": 2, "admission_date": "2024-01-08", "discharge_date": "2024-01-06", "diagnosis": "cold"},  # Invalid
            {"patient_id": 3, "admission_date": "2024-01-10", "discharge_date": "2024-01-15", "diagnosis": "pneumonia"},
            {"patient_id": 4, "admission_date": "2024-01-12", "discharge_date": "2024-01-10", "diagnosis": "bronchitis"},  # Invalid
            {"patient_id": 5, "admission_date": "2024-01-15", "discharge_date": "2024-01-20", "diagnosis": "asthma"},
            {"patient_id": 6, "admission_date": "2024-01-18", "discharge_date": "2024-01-25", "diagnosis": "copd"},
        ],
    }
    
    if task_id not in datasets:
        raise ValueError(f"Unknown task_id: {task_id}")
    
    return copy.deepcopy(datasets[task_id])


def get_target_dataset(task_id: str) -> List[Dict[str, Any]]:
    """
    Returns the target (clean) dataset for each task.
    This is the "perfect" state the agent should achieve.
    """
    targets = {
        "triage-clean-easy": [
            {"patient_id": 1, "age": 25, "diagnosis": "flu", "admission_date": "2024-01-01"},
            {"patient_id": 2, "age": 0, "diagnosis": "cold", "admission_date": "2024-01-02"},  # Filled null
            {"patient_id": 3, "age": 45, "diagnosis": "pneumonia", "admission_date": "2024-01-03"},
            {"patient_id": 4, "age": 0, "diagnosis": "bronchitis", "admission_date": "2024-01-04"},  # Filled null
            {"patient_id": 5, "age": 60, "diagnosis": "asthma", "admission_date": "2024-01-05"},
        ],
        "triage-clean-medium": [
            {"patient_id": 1, "age": 25, "diagnosis": "flu", "severity": "low"},
            {"patient_id": 2, "age": 30, "diagnosis": "cold", "severity": "low"},
            {"patient_id": 3, "age": 45, "diagnosis": "pneumonia", "severity": "high"},
            {"patient_id": 4, "age": 55, "diagnosis": "bronchitis", "severity": "medium"},
            {"patient_id": 5, "age": 60, "diagnosis": "asthma", "severity": "medium"},
            {"patient_id": 6, "age": 70, "diagnosis": "copd", "severity": "high"},
        ],
        "triage-clean-hard": [
            {"patient_id": 1, "admission_date": "2024-01-05", "discharge_date": "2024-01-10", "diagnosis": "flu"},
            {"patient_id": 3, "admission_date": "2024-01-10", "discharge_date": "2024-01-15", "diagnosis": "pneumonia"},
            {"patient_id": 5, "admission_date": "2024-01-15", "discharge_date": "2024-01-20", "diagnosis": "asthma"},
            {"patient_id": 6, "admission_date": "2024-01-18", "discharge_date": "2024-01-25", "diagnosis": "copd"},
        ],
    }
    
    if task_id not in targets:
        raise ValueError(f"Unknown task_id: {task_id}")
    
    return copy.deepcopy(targets[task_id])


def detect_schema_errors(dataset: List[Dict[str, Any]], task_id: str) -> List[str]:
    """
    Detects data quality issues in the current dataset.
    Returns hints for the agent.
    """
    errors = []
    
    if not dataset:
        return errors
    
    # Check for null values
    columns = list(dataset[0].keys())
    for col in columns:
        null_count = sum(1 for row in dataset if row.get(col) is None)
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} null value(s)")
    
    # Check for duplicates
    seen = set()
    duplicates = 0
    for row in dataset:
        row_tuple = tuple(sorted(row.items()))
        if row_tuple in seen:
            duplicates += 1
        seen.add(row_tuple)
    
    if duplicates > 0:
        errors.append(f"{duplicates} duplicate row(s) detected")
    
    # Check for non-numeric ages (medium task)
    if task_id == "triage-clean-medium":
        string_ages = sum(1 for row in dataset if isinstance(row.get("age"), str))
        if string_ages > 0:
            errors.append(f"Column 'age' has {string_ages} non-numeric value(s)")
    
    # Check for logical errors (hard task)
    if task_id == "triage-clean-hard":
        invalid_rows = 0
        for row in dataset:
            admission = row.get("admission_date", "")
            discharge = row.get("discharge_date", "")
            if admission and discharge and discharge < admission:
                invalid_rows += 1
        
        if invalid_rows > 0:
            errors.append(f"{invalid_rows} row(s) with discharge_date before admission_date")
    
    return errors


def dataset_to_string(dataset: List[Dict[str, Any]]) -> str:
    """
    Converts dataset to a stringified JSON format.
    """
    return json.dumps(dataset, indent=2)


def calculate_reward(current_dataset: List[Dict[str, Any]], target_dataset: List[Dict[str, Any]]) -> tuple:
    """
    Calculates partial reward by comparing current dataset to target.
    Returns (reward, accuracy, correct_cells, total_cells).
    
    Reward is strictly clamped between 0.05 and 0.95.
    """
    if not target_dataset:
        return 0.05, 0.0, 0, 0
    
    # Convert datasets to comparable format
    current_set = set()
    for row in current_dataset:
        row_tuple = tuple(sorted(row.items()))
        current_set.add(row_tuple)
    
    target_set = set()
    for row in target_dataset:
        row_tuple = tuple(sorted(row.items()))
        target_set.add(row_tuple)
    
    # Calculate cell-level accuracy
    total_cells = len(target_dataset) * len(target_dataset[0]) if target_dataset else 1
    
    # Count matching rows
    matching_rows = len(current_set & target_set)
    total_target_rows = len(target_set)
    
    # Calculate correct cells (approximation based on matching rows)
    if total_target_rows > 0:
        correct_cells = matching_rows * len(target_dataset[0]) if target_dataset else 0
    else:
        correct_cells = 0
    
    # Calculate accuracy
    accuracy = correct_cells / total_cells if total_cells > 0 else 0.0
    
    # Clamp reward strictly between 0.05 and 0.95
    reward = max(0.05, min(0.95, accuracy))
    
    return reward, accuracy, correct_cells, total_cells


def apply_action(dataset: List[Dict[str, Any]], action: Action, task_id: str) -> List[Dict[str, Any]]:
    """
    Applies the specified action to the dataset.
    Returns a new modified dataset.
    """
    new_dataset = copy.deepcopy(dataset)
    
    action_type = action.action_type
    params = action.parameters
    
    if action_type == "DROP_NULLS":
        column = params.get("column")
        if column:
            new_dataset = [row for row in new_dataset if row.get(column) is not None]
        else:
            # Drop all rows with any null values
            new_dataset = [
                row for row in new_dataset
                if all(v is not None for v in row.values())
            ]
    
    elif action_type == "FILL_NULLS":
        column = params.get("column")
        value = params.get("value", 0)
        if column:
            for row in new_dataset:
                if row.get(column) is None:
                    row[column] = value
    
    elif action_type == "REPLACE_VALUE":
        column = params.get("column")
        old_value = params.get("old_value")
        new_value = params.get("new_value")
        
        if column and old_value is not None and new_value is not None:
            for row in new_dataset:
                if row.get(column) == old_value:
                    row[column] = new_value
    
    elif action_type == "DROP_DUPLICATES":
        seen = set()
        unique_dataset = []
        for row in new_dataset:
            row_tuple = tuple(sorted(row.items()))
            if row_tuple not in seen:
                seen.add(row_tuple)
                unique_dataset.append(row)
        new_dataset = unique_dataset
    
    elif action_type == "DROP_INVALID_ROWS":
        # Task-specific logic
        if task_id == "triage-clean-hard":
            # Drop rows where discharge_date < admission_date
            valid_rows = []
            for row in new_dataset:
                admission = row.get("admission_date", "")
                discharge = row.get("discharge_date", "")
                if not (admission and discharge and discharge < admission):
                    valid_rows.append(row)
            new_dataset = valid_rows
    
    return new_dataset


def create_observation(dataset: List[Dict[str, Any]], task_id: str) -> Observation:
    """
    Creates an observation from the current dataset state.
    """
    patient_data = dataset_to_string(dataset)
    schema_errors = detect_schema_errors(dataset, task_id)
    
    return Observation(
        patient_data=patient_data,
        schema_errors=schema_errors,
        available_actions=AVAILABLE_ACTIONS
    )


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest):
    """
    Resets the environment to the initial state for the specified task.
    """
    task_id = request.task_id
    
    try:
        initial_dataset = get_initial_dataset(task_id)
        target_dataset = get_target_dataset(task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Initialize session state
    SESSIONS[task_id] = {
        "current_dataset": initial_dataset,
        "target_dataset": target_dataset,
        "step_count": 0,
        "task_id": task_id
    }
    
    observation = create_observation(initial_dataset, task_id)
    
    return ResetResponse(
        observation=observation,
        task_id=task_id
    )


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    """
    Executes one step in the environment with the given action.
    Returns observation, reward, done, and info.
    """
    task_id = request.task_id
    action = request.action
    
    if task_id not in SESSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Task '{task_id}' not initialized. Call /reset first."
        )
    
    session = SESSIONS[task_id]
    current_dataset = session["current_dataset"]
    target_dataset = session["target_dataset"]
    
    # Apply action
    try:
        new_dataset = apply_action(current_dataset, action, task_id)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error applying action: {str(e)}"
        )
    
    # Update session
    session["current_dataset"] = new_dataset
    session["step_count"] += 1
    
    # Calculate reward
    reward, accuracy, correct_cells, total_cells = calculate_reward(new_dataset, target_dataset)
    
    # Create observation
    observation = create_observation(new_dataset, task_id)
    
    # Check if done (perfect match or no more errors)
    done = len(observation.schema_errors) == 0 or reward >= 0.95
    
    info = {
        "accuracy": accuracy,
        "correct_cells": correct_cells,
        "total_cells": total_cells,
        "step_count": session["step_count"],
        "action_applied": action.action_type
    }
    
    return StepResponse(
        observation=observation,
        reward=reward,
        done=done,
        info=info
    )


@app.post("/state", response_model=StateResponse)
async def get_state(request: StateRequest):
    """
    Returns the current state of the environment without taking an action.
    """
    task_id = request.task_id
    
    if task_id not in SESSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Task '{task_id}' not initialized. Call /reset first."
        )
    
    session = SESSIONS[task_id]
    current_dataset = session["current_dataset"]
    target_dataset = session["target_dataset"]
    
    # Calculate current reward
    reward, _, _, _ = calculate_reward(current_dataset, target_dataset)
    
    observation = create_observation(current_dataset, task_id)
    
    return StateResponse(
        observation=observation,
        current_reward=reward,
        step_count=session["step_count"]
    )


@app.get("/")
async def root():
    """
    Health check endpoint.
    """
    return {
        "message": "DataCleaner-Env API is running",
        "version": "1.0.0",
        "available_tasks": [
            "triage-clean-easy",
            "triage-clean-medium",
            "triage-clean-hard"
        ]
    }


@app.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)