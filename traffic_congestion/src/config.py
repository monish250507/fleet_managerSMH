from pathlib import Path

# Project configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

# File paths
RAW_DATASET_PATH = RAW_DATA_DIR / "smh_dataset.csv"
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "smh_dataset_engineered.csv"
LABELING_METADATA_PATH = MODELS_DIR / "labeling_metadata.json"
TRAINED_MODEL_PATH = MODELS_DIR / "trained_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Zone congestion classification thresholds
SAFE_THRESHOLD = {
    'task_to_agv_ratio': 0.9,
    'avg_zone_speed': 1.1,
    'path_overlap_score': 0.25,
    'avg_zone_wait_time': 8
}

WARNING_CONDITIONS = {
    'task_to_agv_ratio_range': (0.9, 1.3),      # (0.9, 1.3]
    'avg_zone_speed_range': (0.7, 1.1),          # [0.7, 1.1)
    'path_overlap_score_range': (0.25, 0.55),    # (0.25, 0.55]
    'avg_zone_wait_time_range': (8, 25),         # (8, 25]
    'zone_agv_count_threshold': 4
}

CRITICAL_CONDITIONS = {
    'task_to_agv_ratio': 1.3,
    'avg_zone_speed': 0.7,
    'path_overlap_score': 0.55,
    'min_zone_agv_count_for_path_conflict': 5,
    'avg_zone_wait_time': 25
}