# AGV Zone Congestion Classifier

A production-quality Python project that implements a THREE-STATE zone congestion classifier (SAFE / WARNING / CRITICAL) using rule-based labeling + multinomial logistic regression for AGV fleet control systems.

## Project Goal

Predict zone operational state from snapshot metrics and provide actionable early-warning (WARNING) signals for AGV fleet control.

## State Definitions

### Derived Signals
- `task_to_agv_ratio` = task_arrival_rate / max(zone_agv_count, 1)
- `speed_drop_factor` = 1 / max(avg_zone_speed, 0.01)
- `spatial_conflict_index` = path_overlap_score * zone_agv_count
- `agv_density_pressure` = zone_agv_count * zone_density
- `wait_time_pressure` = avg_zone_wait_time * task_arrival_rate
- `congestion_pressure_score` = 0.30*agv_density_pressure + 0.25*task_to_agv_ratio + 0.20*speed_drop_factor + 0.15*spatial_conflict_index + 0.10*wait_time_pressure

### Classification Rules

**SAFE (label 0)** — ALL must hold:
- task_to_agv_ratio ≤ 0.9
- avg_zone_speed ≥ 1.1
- path_overlap_score ≤ 0.25
- avg_zone_wait_time ≤ 8

**WARNING (label 1)** — ANY TWO OR MORE:
- task_to_agv_ratio ∈ (0.9, 1.3]
- avg_zone_speed ∈ [0.7, 1.1)
- path_overlap_score ∈ (0.25, 0.55]
- avg_zone_wait_time ∈ (8, 25]
- zone_agv_count ≥ 4

**CRITICAL (label 2)** — ANY ONE:
- task_to_agv_ratio ≥ 1.3
- avg_zone_speed ≤ 0.7
- (path_overlap_score ≥ 0.55 AND zone_agv_count ≥ 5)
- avg_zone_wait_time ≥ 25
- congestion_pressure_score ≥ 75th percentile (computed from training data)

## Project Structure

```
├── data/
│   ├── raw/
│   │   └── smh_dataset.csv
│   └── processed/
│       └── smh_dataset_engineered.csv
├── src/
│   ├── config.py
│   ├── feature_engineering/
│   │   └── build_features.py
│   ├── labels/
│   │   └── label_generator.py
│   ├── models/
│   │   ├── train_model.py
│   │   └── inference.py
│   └── utils/
│       └── validation.py
├── artifacts/
│   └── models/
│       ├── trained_model.pkl
│       ├── scaler.pkl
│       └── labeling_metadata.json
├── requirements.txt
└── README.md
```

## Installation

1. Activate the virtual environment:
```bash
# On Windows
.venv\Scripts\activate

# On Unix/Mac
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Feature Engineering
Process the raw dataset to create derived features:
```bash
python -m src.feature_engineering.build_features
```

### 2. Label Generation
Generate labels based on the classification rules:
```bash
python -m src.labels.label_generator
```

### 3. Model Training
Train the multinomial logistic regression model:
```bash
python -m src.models.train_model
```

### 4. Validation
Validate the datasets and model inputs:
```bash
python -m src.utils.validation
```

### 5. Inference
Use the trained model for predictions:
```bash
python -m src.models.inference
```

## Model Configuration

- Algorithm: Multinomial Logistic Regression
- Solver: SAGA
- Class handling: Multinomial
- Class weighting: Balanced
- Cross-validation: 5-fold Stratified
- Metrics: Macro-F1 and per-class recall

## Example Usage

```python
from src.models.inference import ZoneCongestionPredictor

# Initialize predictor
predictor = ZoneCongestionPredictor()

# Example zone snapshot
example_snapshot = {
    'task_arrival_rate': 5.0,
    'zone_agv_count': 3,
    'avg_zone_speed': 1.0,
    'path_overlap_score': 0.3,
    'zone_density': 0.6,
    'avg_zone_wait_time': 15.0
}

# Make prediction
result = predictor.predict(example_snapshot)
print(f"State: {result['state']}")
print(f"Probabilities: {result['probabilities']}")
print(f"Dominant Signals: {result['dominant_signals']}")
```

## Output Format

The inference module returns a dictionary with:
- `state`: "SAFE" | "WARNING" | "CRITICAL"
- `probabilities`: Dictionary with probabilities for each state
- `dominant_signals`: List of the most influential features for the prediction

## Requirements

- Python 3.9+
- Dependencies listed in requirements.txt
- pathlib for path handling
- No external services required