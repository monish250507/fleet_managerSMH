"""
Feature engineering module for AGV zone congestion classification.
Implements derived signals as specified in the requirements.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATASET_PATH, PROCESSED_DATASET_PATH


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build derived features from raw AGV zone data.
    
    Args:
        df: Raw dataset with columns: task_arrival_rate, zone_agv_count, 
            avg_zone_speed, path_overlap_score, zone_density, avg_zone_wait_time
    
    Returns:
        DataFrame with original features plus derived signals
    """
    # Create a copy to avoid modifying original
    df_features = df.copy()
    
    # Derived signals as specified
    df_features['task_to_agv_ratio'] = df_features['task_arrival_rate'] / np.maximum(df_features['zone_agv_count'], 1)
    df_features['speed_drop_factor'] = 1 / np.maximum(df_features['avg_zone_speed'], 0.01)
    df_features['spatial_conflict_index'] = df_features['path_overlap_score'] * df_features['zone_agv_count']
    df_features['agv_density_pressure'] = df_features['zone_agv_count'] * df_features['zone_density']
    df_features['wait_time_pressure'] = df_features['avg_zone_wait_time'] * df_features['task_arrival_rate']
    
    # Congestion pressure score
    df_features['congestion_pressure_score'] = (
        0.30 * df_features['agv_density_pressure'] +
        0.25 * df_features['task_to_agv_ratio'] +
        0.20 * df_features['speed_drop_factor'] +
        0.15 * df_features['spatial_conflict_index'] +
        0.10 * df_features['wait_time_pressure']
    )
    
    return df_features


def main():
    """Main function to run feature engineering pipeline."""
    print("Loading raw dataset...")
    df = pd.read_csv(RAW_DATASET_PATH)
    
    print("Building features...")
    df_engineered = build_features(df)
    
    print("Saving processed dataset...")
    PROCESSED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_engineered.to_csv(PROCESSED_DATASET_PATH, index=False)
    
    print(f"Feature engineering complete. Processed dataset saved to {PROCESSED_DATASET_PATH}")
    print(f"Dataset shape: {df_engineered.shape}")
    print(f"Columns: {list(df_engineered.columns)}")


if __name__ == "__main__":
    main()