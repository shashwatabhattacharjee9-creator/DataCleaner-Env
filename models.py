"""
Pydantic models for DataCleaner-Env - A stateful data cleaning environment.
Built for Meta PyTorch Hackathon.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class Observation(BaseModel):
    """
    Observation returned to the agent after each step.
    Contains the current state of the dataset and hints.
    """
    patient_data: str = Field(
        ..., 
        description="Stringified JSON or markdown table of current dataset state"
    )
    schema_errors: List[str] = Field(
        default_factory=list,
        description="Hints about data quality issues (e.g., 'Column age has 3 null values')"
    )
    available_actions: List[str] = Field(
        default_factory=list,
        description="List of available action types the agent can take"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_data": '[{"patient_id": 1, "age": null, "diagnosis": "flu"}]',
                "schema_errors": ["Column 'age' has 1 null values", "2 duplicate rows detected"],
                "available_actions": ["DROP_NULLS", "FILL_NULLS", "DROP_DUPLICATES"]
            }
        }


class Action(BaseModel):
    """
    Action submitted by the agent to modify the dataset.
    """
    action_type: str = Field(
        ...,
        description="Type of cleaning action (e.g., DROP_NULLS, FILL_NULLS, REPLACE_VALUE, DROP_DUPLICATES, DROP_INVALID_ROWS)"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the action (e.g., {'column': 'age', 'value': 0})"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "action_type": "FILL_NULLS",
                "parameters": {"column": "age", "value": 0}
            }
        }


class Reward(BaseModel):
    """
    Reward returned after each step.
    Partial rewards based on similarity to target dataset.
    """
    reward: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Partial reward value between 0.0 and 1.0"
    )
    done: bool = Field(
        ...,
        description="Whether the episode is complete"
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional information (e.g., accuracy metrics, error messages)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "reward": 0.75,
                "done": False,
                "info": {"accuracy": 0.75, "correct_cells": 15, "total_cells": 20}
            }
        }


class ResetRequest(BaseModel):
    """
    Request to reset the environment to a specific task.
    """
    task_id: str = Field(
        ...,
        description="Task identifier (e.g., 'triage-clean-easy', 'triage-clean-medium', 'triage-clean-hard')"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "triage-clean-easy"
            }
        }


class ResetResponse(BaseModel):
    """
    Response from reset endpoint containing initial observation.
    """
    observation: Observation
    task_id: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "observation": {
                    "patient_data": '[{"patient_id": 1, "age": 25}]',
                    "schema_errors": [],
                    "available_actions": ["DROP_NULLS", "FILL_NULLS"]
                },
                "task_id": "triage-clean-easy"
            }
        }


class StepRequest(BaseModel):
    """
    Request to take a step in the environment.
    """
    task_id: str = Field(
        ...,
        description="Task identifier for the current session"
    )
    action: Action = Field(
        ...,
        description="Action to execute"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "triage-clean-easy",
                "action": {
                    "action_type": "DROP_DUPLICATES",
                    "parameters": {}
                }
            }
        }


class StepResponse(BaseModel):
    """
    Response from step endpoint.
    """
    observation: Observation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "observation": {
                    "patient_data": '[{"patient_id": 1, "age": 25}]',
                    "schema_errors": [],
                    "available_actions": ["DROP_NULLS"]
                },
                "reward": 0.85,
                "done": False,
                "info": {"accuracy": 0.85}
            }
        }


class StateRequest(BaseModel):
    """
    Request to get the current state of the environment.
    """
    task_id: str = Field(
        ...,
        description="Task identifier for the current session"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "triage-clean-easy"
            }
        }


class StateResponse(BaseModel):
    """
    Response from state endpoint.
    """
    observation: Observation
    current_reward: float = Field(ge=0.0, le=1.0)
    step_count: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "observation": {
                    "patient_data": '[{"patient_id": 1, "age": 25}]',
                    "schema_errors": [],
                    "available_actions": ["DROP_NULLS"]
                },
                "current_reward": 0.75,
                "step_count": 3
            }
        }