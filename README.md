---
title: DataCleaner-Env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# DataCleaner-Env 🧹

**A State-of-the-Art OpenEnv Environment for Data Cleaning with Partial Rewards**

# DataCleaner-Env 🧹

**A State-of-the-Art OpenEnv Environment for Data Cleaning with Partial Rewards**

Built for the Meta PyTorch Hackathon.

## 🎯 Overview

DataCleaner-Env is a stateful reinforcement learning environment that simulates real-world data cleaning tasks. AI agents interact with corrupted datasets and receive partial rewards based on how closely their cleaned data matches a hidden "perfect" dataset.

### Key Features

- ✅ **Stateful Sessions**: Maintains dataset state across multiple agent actions
- ✅ **Partial Rewards**: Granular feedback (0.05 - 0.95) based on cell-level accuracy
- ✅ **Three Difficulty Levels**: Easy, Medium, and Hard cleaning tasks
- ✅ **RESTful API**: Standard OpenAI Gym-style endpoints (`/reset`, `/step`, `/state`)
- ✅ **Production-Ready**: Type-safe Pydantic models, comprehensive error handling

## 📋 Task Descriptions

### Task 1: `triage-clean-easy`
**Challenge**: Duplicate rows and null values  
**Goal**: Remove duplicates and fill null ages with 0

**Initial Dataset Issues**:
- 2 duplicate patient records
- 2 null values in the `age` column

**Required Actions**:
1. `DROP_DUPLICATES` to remove duplicate rows
2. `FILL_NULLS` (column: "age", value: 0) to impute missing ages

### Task 2: `triage-clean-medium`
**Challenge**: Mixed data types (string ages instead of numeric)  
**Goal**: Convert string age values to integers

**Initial Dataset Issues**:
- Ages like "twenty-five", "forty-five", "sixty" instead of 25, 45, 60

**Required Actions**:
1. `REPLACE_VALUE` (old: "twenty-five", new: 25)
2. `REPLACE_VALUE` (old: "forty-five", new: 45)
3. `REPLACE_VALUE` (old: "sixty", new: 60)

### Task 3: `triage-clean-hard`
**Challenge**: Logical inconsistencies  
**Goal**: Remove rows where discharge_date occurs before admission_date

**Initial Dataset Issues**:
- 2 patients with impossible date sequences

**Required Actions**:
1. `DROP_INVALID_ROWS` to remove logically inconsistent records

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd datacleaner-env

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
# Start the FastAPI server
python server/app.py
```

The server will start on `http://localhost:8000`

### Testing the Environment

```bash
# In a separate terminal, run the test client
python test_client.py
```

## 📡 API Endpoints

### 1. Health Check
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy"
}
```

### 2. Reset Environment
```http
POST /reset
```

**Request Body**:
```json
{
  "task_id": "triage-clean-easy"
}
```

**Response**:
```json
{
  "observation": {
    "patient_data": "[{...}, {...}]",
    "schema_errors": ["Column 'age' has 2 null value(s)", "2 duplicate row(s) detected"],
    "available_actions": ["DROP_NULLS", "FILL_NULLS", "REPLACE_VALUE", "DROP_DUPLICATES", "DROP_INVALID_ROWS"]
  },
  "task_id": "triage-clean-easy"
}
```

### 3. Take Step
```http
POST /step
```

**Request Body**:
```json
{
  "task_id": "triage-clean-easy",
  "action": {
    "action_type": "DROP_DUPLICATES",
    "parameters": {}
  }
}
```

**Response**:
```json
{
  "observation": {
    "patient_data": "[{...}]",
    "schema_errors": ["Column 'age' has 2 null value(s)"],
    "available_actions": ["DROP_NULLS", "FILL_NULLS", ...]
  },
  "reward": 0.65,
  "done": false,
  "info": {
    "accuracy": 0.6,
    "correct_cells": 12,
    "total_cells": 20,
    "step_count": 1,
    "action_applied": "DROP_DUPLICATES"
  }
}
```

### 4. Get Current State
```http
POST /state
```

**Request Body**:
```json
{
  "task_id": "triage-clean-easy"
}
```

**Response**:
```json
{
  "observation": {...},
  "current_reward": 0.65,
  "step_count": 1
}
```

## 🎮 Available Actions

| Action Type | Description | Parameters |
|-------------|-------------|------------|
| `DROP_NULLS` | Remove rows with null values | `{"column": "age"}` (optional) |
| `FILL_NULLS` | Fill null values with a constant | `{"column": "age", "value": 0}` |
| `REPLACE_VALUE` | Replace specific values | `{"column": "age", "old_value": "twenty-five", "new_value": 25}` |
| `DROP_DUPLICATES` | Remove duplicate rows | `{}` |
| `DROP_INVALID_ROWS` | Remove logically invalid rows | `{}` (task-specific logic) |

## 📊 Reward System

The environment uses a **partial reward** system that provides granular feedback:

### Calculation Method
```python
reward = (matching_cells / total_target_cells)
reward = clamp(reward, min=0.05, max=0.95)
```

### Reward Interpretation
- `0.05 - 0.30`: Poor match, significant cleaning needed
- `0.31 - 0.60`: Moderate progress, some issues remain
- `0.61 - 0.85`: Good progress, minor issues
- `0.86 - 0.95`: Excellent match, nearly perfect

**Note**: Rewards are **strictly clamped** between 0.05 and 0.95 to avoid binary outcomes.

## 🏗️ Architecture

### File Structure
```
datacleaner-env/
├── models.py           # Pydantic data models
├── server/
│   └── app.py         # FastAPI server implementation
├── test_client.py     # Test suite and examples
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

### State Management

The server maintains a global `SESSIONS` dictionary:

```python
SESSIONS = {
    "triage-clean-easy": {
        "current_dataset": [...],  # Current state
        "target_dataset": [...],   # Hidden target
        "step_count": 0,
        "task_id": "triage-clean-easy"
    }
}
```

### Data Flow

```
Agent → /reset → Initialize Session → Return Observation
    ↓
Agent → /step → Apply Action → Calculate Reward → Return (Obs, Reward, Done, Info)
    ↓
Agent → /state → Return Current State (no modification)
```

## 🧪 Example Usage

### Python Client

```python
import requests

BASE_URL = "http://localhost:8000"

# Reset environment
response = requests.post(
    f"{BASE_URL}/reset",
    json={"task_id": "triage-clean-easy"}
)
obs = response.json()["observation"]

# Take action
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

result = response.json()
print(f"Reward: {result['reward']}")
print(f"Done: {result['done']}")
print(f"Info: {result['info']}")
```

### cURL

```bash
# Reset
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "triage-clean-easy"}'

# Step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "triage-clean-easy",
    "action": {
      "action_type": "DROP_DUPLICATES",
      "parameters": {}
    }
  }'
```

## 🔬 Advanced Features

### Schema Error Detection

The environment automatically detects and reports:
- Null values per column
- Duplicate rows
- Type inconsistencies (e.g., string ages)
- Logical errors (e.g., invalid date sequences)

### Type Safety

All API models use Pydantic for runtime validation:
- `Observation`: Current state and hints
- `Action`: Agent commands
- `Reward`: Feedback metrics
- `ResetRequest/Response`: Session initialization
- `StepRequest/Response`: Action execution
- `StateRequest/Response`: State queries

## 🎓 Training Tips

1. **Start with Easy**: Master Task 1 before attempting harder tasks
2. **Use Schema Errors**: The `schema_errors` field provides valuable hints
3. **Monitor Rewards**: Track reward progression to gauge cleaning effectiveness
4. **Iterative Approach**: Complex tasks may require multiple actions
5. **Check State**: Use `/state` endpoint to inspect current progress

## 🛠️ Development

### Running Tests

```bash
# Start the server in one terminal
python server/app.py

# Run tests in another terminal
python test_client.py
```

### Adding New Tasks

1. Add initial dataset to `get_initial_dataset()`
2. Add target dataset to `get_target_dataset()`
3. Update `detect_schema_errors()` for task-specific hints
4. Implement task-specific logic in `apply_action()`

## 📝 License

This project is built for educational purposes as part of the Meta PyTorch Hackathon.

## 🙏 Acknowledgments

Built with:
- FastAPI for the REST API
- Pydantic for data validation
- Uvicorn for ASGI serving

---

**Ready to train your data cleaning agent? Start the server and let's clean some data! 🧹✨**