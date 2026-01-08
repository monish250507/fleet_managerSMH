"""
Validation utilities for AGV zone congestion classification project.
Provides data validation and consistency checks.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATASET_PATH, RAW_DATASET_PATH


def validate_raw_data(df: pd.DataFrame) -> dict:
    """
    Validate raw AGV zone data.
    
    Args:
        df: Raw dataset
        
    Returns:
        dict: Validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    required_columns = [
        'task_arrival_rate', 'zone_agv_count', 'avg_zone_speed', 
        'path_overlap_score', 'zone_density', 'avg_zone_wait_time'
    ]
    
    # Check for required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        results['errors'].append(f"Missing required columns: {missing_columns}")
        results['valid'] = False
    else:
        results['info'].append(f"All required columns present: {required_columns}")
    
    # Check for non-null values in required columns
    for col in required_columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            results['errors'].append(f"Column '{col}' has {null_count} null values")
            results['valid'] = False
    
    # Check for reasonable ranges in numeric data
    if 'task_arrival_rate' in df.columns:
        if (df['task_arrival_rate'] < 0).any():
            results['errors'].append("task_arrival_rate has negative values")
            results['valid'] = False
    
    if 'zone_agv_count' in df.columns:
        if (df['zone_agv_count'] < 0).any():
            results['errors'].append("zone_agv_count has negative values")
            results['valid'] = False
    
    if 'avg_zone_speed' in df.columns:
        if (df['avg_zone_speed'] < 0).any():
            results['errors'].append("avg_zone_speed has negative values")
            results['valid'] = False
    
    if 'path_overlap_score' in df.columns:
        if (df['path_overlap_score'] < 0).any() or (df['path_overlap_score'] > 1).any():
            results['warnings'].append("path_overlap_score has values outside [0, 1] range")
    
    if 'zone_density' in df.columns:
        if (df['zone_density'] < 0).any() or (df['zone_density'] > 1).any():
            results['warnings'].append("zone_density has values outside [0, 1] range")
    
    if 'avg_zone_wait_time' in df.columns:
        if (df['avg_zone_wait_time'] < 0).any():
            results['errors'].append("avg_zone_wait_time has negative values")
            results['valid'] = False
    
    return results


def validate_processed_data(df: pd.DataFrame) -> dict:
    """
    Validate processed AGV zone data with engineered features and labels.
    
    Args:
        df: Processed dataset
        
    Returns:
        dict: Validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    # Check if label column exists
    if 'label' not in df.columns:
        results['errors'].append("Missing 'label' column")
        results['valid'] = False
    else:
        results['info'].append("Label column present")
        
        # Check if labels are in the expected range (0, 1, 2)
        unique_labels = set(df['label'].unique())
        expected_labels = {0, 1, 2}
        if not unique_labels.issubset(expected_labels):
            results['errors'].append(f"Unexpected label values: {unique_labels - expected_labels}")
            results['valid'] = False
        else:
            results['info'].append(f"Valid label values found: {sorted(unique_labels)}")
    
    # Check for derived features
    expected_features = [
        'task_to_agv_ratio', 'speed_drop_factor', 'spatial_conflict_index', 
        'agv_density_pressure', 'wait_time_pressure', 'congestion_pressure_score'
    ]
    
    missing_features = set(expected_features) - set(df.columns)
    if missing_features:
        results['errors'].append(f"Missing derived features: {missing_features}")
        results['valid'] = False
    else:
        results['info'].append(f"All derived features present: {expected_features}")
    
    # Check for infinite or NaN values in features
    feature_cols = [col for col in df.columns if col != 'label']
    for col in feature_cols:
        # Only check numeric columns for NaN and infinite values
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().any():
                results['errors'].append(f"Column '{col}' has NaN values")
                results['valid'] = False
            if np.isinf(df[col]).any():
                results['errors'].append(f"Column '{col}' has infinite values")
                results['valid'] = False
    
    return results


def validate_model_input(snapshot: dict) -> dict:
    """
    Validate a single zone snapshot for model input.
    
    Args:
        snapshot: Dictionary representing a zone snapshot
        
    Returns:
        dict: Validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    required_fields = [
        'task_arrival_rate', 'zone_agv_count', 'avg_zone_speed', 
        'path_overlap_score', 'zone_density', 'avg_zone_wait_time'
    ]
    
    # Check for required fields
    missing_fields = set(required_fields) - set(snapshot.keys())
    if missing_fields:
        results['errors'].append(f"Missing required fields: {missing_fields}")
        results['valid'] = False
    
    # Validate numeric values
    for field in required_fields:
        if field in snapshot:
            value = snapshot[field]
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                results['errors'].append(f"Field '{field}' has invalid value: {value}")
                results['valid'] = False
            elif field in ['task_arrival_rate', 'zone_agv_count', 'avg_zone_wait_time'] and value < 0:
                results['errors'].append(f"Field '{field}' has negative value: {value}")
                results['valid'] = False
            elif field in ['avg_zone_speed'] and value < 0:
                results['errors'].append(f"Field '{field}' has negative value: {value}")
                results['valid'] = False
            elif field in ['path_overlap_score', 'zone_density'] and (value < 0 or value > 1):
                results['warnings'].append(f"Field '{field}' has value outside expected range [0, 1]: {value}")
    
    return results


def main():
    """Main function to run validation checks."""
    print("Validating raw dataset...")
    raw_df = pd.read_csv(RAW_DATASET_PATH)
    raw_validation = validate_raw_data(raw_df)
    
    print("\nRaw Data Validation Results:")
    print(f"Valid: {raw_validation['valid']}")
    if raw_validation['errors']:
        print(f"Errors: {raw_validation['errors']}")
    if raw_validation['warnings']:
        print(f"Warnings: {raw_validation['warnings']}")
    if raw_validation['info']:
        print(f"Info: {raw_validation['info']}")
    
    print("\nValidating processed dataset...")
    try:
        processed_df = pd.read_csv(PROCESSED_DATASET_PATH)
        processed_validation = validate_processed_data(processed_df)
        
        print("\nProcessed Data Validation Results:")
        print(f"Valid: {processed_validation['valid']}")
        if processed_validation['errors']:
            print(f"Errors: {processed_validation['errors']}")
        if processed_validation['warnings']:
            print(f"Warnings: {processed_validation['warnings']}")
        if processed_validation['info']:
            print(f"Info: {processed_validation['info']}")
    except FileNotFoundError:
        print(f"Processed dataset not found at {PROCESSED_DATASET_PATH}")
        print("This is expected if feature engineering and labeling have not been run yet.")


if __name__ == "__main__":
    main()