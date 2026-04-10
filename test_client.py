"""
Test script for DataCleaner-Env API.
Demonstrates all three tasks with agent interactions.
"""

import json
import requests
from typing import Dict, Any


BASE_URL = "http://localhost:8000"


def print_observation(obs: Dict[str, Any], title: str = "Observation"):
    """Pretty print observation."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print("\nCurrent Dataset:")
    data = json.loads(obs["patient_data"])
    for i, row in enumerate(data, 1):
        print(f"  {i}. {row}")
    
    print(f"\nSchema Errors: {obs['schema_errors']}")
    print(f"Available Actions: {obs['available_actions']}")


def test_task_easy():
    """Test Task 1: Drop duplicates and fill nulls."""
    print("\n" + "="*60)
    print("TASK 1: TRIAGE-CLEAN-EASY")
    print("Goal: Remove duplicates and fill null ages with 0")
    print("="*60)
    
    # Reset
    response = requests.post(f"{BASE_URL}/reset", json={"task_id": "triage-clean-easy"})
    data = response.json()
    print_observation(data["observation"], "Initial State")
    
    # Step 1: Drop duplicates
    print("\n>> Action: DROP_DUPLICATES")
    response = requests.post(
        f"{BASE_URL}/step",
        json={
            "task_id": "triage-clean-easy",
            "action": {
                "action_type": "DROP_DUPLICATES",
                "parameters": {}
            }
        }
    )
    data = response.json()
    print(f"Reward: {data['reward']:.3f} | Done: {data['done']}")
    print(f"Info: {data['info']}")
    print_observation(data["observation"], "After Dropping Duplicates")
    
    # Step 2: Fill nulls in age column
    print("\n>> Action: FILL_NULLS (age=0)")
    response = requests.post(
        f"{BASE_URL}/step",
        json={
            "task_id": "triage-clean-easy",
            "action": {
                "action_type": "FILL_NULLS",
                "parameters": {"column": "age", "value": 0}
            }
        }
    )
    data = response.json()
    print(f"Reward: {data['reward']:.3f} | Done: {data['done']}")
    print(f"Info: {data['info']}")
    print_observation(data["observation"], "Final State")


def test_task_medium():
    """Test Task 2: Replace string ages with numeric values."""
    print("\n" + "="*60)
    print("TASK 2: TRIAGE-CLEAN-MEDIUM")
    print("Goal: Replace string ages ('twenty-five') with numeric values")
    print("="*60)
    
    # Reset
    response = requests.post(f"{BASE_URL}/reset", json={"task_id": "triage-clean-medium"})
    data = response.json()
    print_observation(data["observation"], "Initial State")
    
    # Step 1: Replace "twenty-five" with 25
    print("\n>> Action: REPLACE_VALUE (twenty-five -> 25)")
    response = requests.post(
        f"{BASE_URL}/step",
        json={
            "task_id": "triage-clean-medium",
            "action": {
                "action_type": "REPLACE_VALUE",
                "parameters": {
                    "column": "age",
                    "old_value": "twenty-five",
                    "new_value": 25
                }
            }
        }
    )
    data = response.json()
    print(f"Reward: {data['reward']:.3f} | Done: {data['done']}")
    print(f"Info: {data['info']}")
    
    # Step 2: Replace "forty-five" with 45
    print("\n>> Action: REPLACE_VALUE (forty-five -> 45)")
    response = requests.post(
        f"{BASE_URL}/step",
        json={
            "task_id": "triage-clean-medium",
            "action": {
                "action_type": "REPLACE_VALUE",
                "parameters": {
                    "column": "age",
                    "old_value": "forty-five",
                    "new_value": 45
                }
            }
        }
    )
    data = response.json()
    print(f"Reward: {data['reward']:.3f} | Done: {data['done']}")
    
    # Step 3: Replace "sixty" with 60
    print("\n>> Action: REPLACE_VALUE (sixty -> 60)")
    response = requests.post(
        f"{BASE_URL}/step",
        json={
            "task_id": "triage-clean-medium",
            "action": {
                "action_type": "REPLACE_VALUE",
                "parameters": {
                    "column": "age",
                    "old_value": "sixty",
                    "new_value": 60
                }
            }
        }
    )
    data = response.json()
    print(f"Reward: {data['reward']:.3f} | Done: {data['done']}")
    print(f"Info: {data['info']}")
    print_observation(data["observation"], "Final State")


def test_task_hard():
    """Test Task 3: Drop logically invalid rows."""
    print("\n" + "="*60)
    print("TASK 3: TRIAGE-CLEAN-HARD")
    print("Goal: Remove rows where discharge_date < admission_date")
    print("="*60)
    
    # Reset
    response = requests.post(f"{BASE_URL}/reset", json={"task_id": "triage-clean-hard"})
    data = response.json()
    print_observation(data["observation"], "Initial State")
    
    # Step 1: Drop invalid rows
    print("\n>> Action: DROP_INVALID_ROWS")
    response = requests.post(
        f"{BASE_URL}/step",
        json={
            "task_id": "triage-clean-hard",
            "action": {
                "action_type": "DROP_INVALID_ROWS",
                "parameters": {}
            }
        }
    )
    data = response.json()
    print(f"Reward: {data['reward']:.3f} | Done: {data['done']}")
    print(f"Info: {data['info']}")
    print_observation(data["observation"], "Final State")


def test_state_endpoint():
    """Test the /state endpoint."""
    print("\n" + "="*60)
    print("TESTING /state ENDPOINT")
    print("="*60)
    
    # First reset a task
    requests.post(f"{BASE_URL}/reset", json={"task_id": "triage-clean-easy"})
    
    # Get state
    response = requests.post(f"{BASE_URL}/state", json={"task_id": "triage-clean-easy"})
    data = response.json()
    
    print(f"\nCurrent Reward: {data['current_reward']:.3f}")
    print(f"Step Count: {data['step_count']}")
    print_observation(data["observation"], "Current State")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("DataCleaner-Env Test Suite")
    print("Make sure the server is running: python server/app.py")
    print("="*60)
    
    try:
        # Test health endpoint
        response = requests.get(f"{BASE_URL}/health")
        print(f"\nServer Health: {response.json()}")
        
        # Run tests
        test_task_easy()
        test_task_medium()
        test_task_hard()
        test_state_endpoint()
        
        print("\n" + "="*60)
        print("All tests completed successfully!")
        print("="*60 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to server.")
        print("Please start the server first: python server/app.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")