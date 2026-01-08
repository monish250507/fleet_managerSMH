"""
Label generator module for AGV zone congestion classification.
Implements SAFE/WARNING/CRITICAL logic exactly as specified.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

# Add src to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATASET_PATH, LABELING_METADATA_PATH


def classify_zone_state(row) -> int:
    """
    Classify zone state based on the specified rules.
    
    Args:
        row: A pandas Series representing a single zone snapshot
        
    Returns:
        int: 0 for SAFE, 1 for WARNING, 2 for CRITICAL
    """
    # Extract values
    task_to_agv_ratio = row['task_to_agv_ratio']
    avg_zone_speed = row['avg_zone_speed']
    path_overlap_score = row['path_overlap_score']
    avg_zone_wait_time = row['avg_zone_wait_time']
    zone_agv_count = row['zone_agv_count']
    congestion_pressure_score = row['congestion_pressure_score']
    
    # Check for CRITICAL state (ANY ONE condition)
    # Using the 75th percentile of congestion_pressure_score from training data
    # This will be computed and passed as a parameter
    critical_conditions = [
        task_to_agv_ratio >= 1.3,
        avg_zone_speed <= 0.7,
        (path_overlap_score >= 0.55 and zone_agv_count >= 5),
        avg_zone_wait_time >= 25,
        congestion_pressure_score >= row.get('congestion_pressure_score_75th_percentile', float('inf'))
    ]
    
    if any(critical_conditions):
        return 2  # CRITICAL
    
    # Check for SAFE state (ALL conditions must hold)
    safe_conditions = [
        task_to_agv_ratio <= 0.9,
        avg_zone_speed >= 1.1,
        path_overlap_score <= 0.25,
        avg_zone_wait_time <= 8
    ]
    
    if all(safe_conditions):
        return 0  # SAFE
    
    # Check for WARNING state (ANY TWO OR MORE conditions)
    warning_conditions = [
        0.9 < task_to_agv_ratio <= 1.3,
        0.7 <= avg_zone_speed < 1.1,
        0.25 < path_overlap_score <= 0.55,
        8 < avg_zone_wait_time <= 25,
        zone_agv_count >= 4
    ]
    
    if sum(warning_conditions) >= 2:
        return 1  # WARNING
    
    # Default to SAFE if no other conditions are met
    return 0  # SAFE


def generate_labels(df: pd.DataFrame, congestion_pressure_75th_percentile: float) -> pd.DataFrame:
    """
    Generate labels for the dataset based on zone state classification rules.
    
    Args:
        df: Processed dataset with engineered features
        congestion_pressure_75th_percentile: 75th percentile of congestion pressure score
        
    Returns:
        DataFrame with added 'label' column
    """
    # Add the 75th percentile as a column for use in classification
    df_with_percentile = df.copy()
    df_with_percentile['congestion_pressure_score_75th_percentile'] = congestion_pressure_75th_percentile
    
    # Apply classification function to each row
    df_with_percentile['label'] = df_with_percentile.apply(classify_zone_state, axis=1)
    
    # Remove the temporary column
    df_result = df_with_percentile.drop(columns=['congestion_pressure_score_75th_percentile'])
    
    return df_result


def main():
    """Main function to run label generation pipeline."""
    print("Loading processed dataset...")
    df = pd.read_csv(PROCESSED_DATASET_PATH)
    
    # Compute the 75th percentile of congestion pressure score for critical condition
    congestion_pressure_75th_percentile = df['congestion_pressure_score'].quantile(0.75)
    
    print("Generating labels...")
    df_labeled = generate_labels(df, congestion_pressure_75th_percentile)
    
    # Save labeling metadata
    metadata = {
        'congestion_pressure_75th_percentile': float(congestion_pressure_75th_percentile),
        'label_distribution': df_labeled['label'].value_counts().to_dict(),
        'label_names': {0: 'SAFE', 1: 'WARNING', 2: 'CRITICAL'}
    }
    
    LABELING_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELING_METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save the labeled dataset back to the same location
    df_labeled.to_csv(PROCESSED_DATASET_PATH, index=False)
    
    print(f"Label generation complete. Metadata saved to {LABELING_METADATA_PATH}")
    print(f"Label distribution: {metadata['label_distribution']}")
    print(f"75th percentile of congestion pressure score: {congestion_pressure_75th_percentile:.2f}")


if __name__ == "__main__":
    main()