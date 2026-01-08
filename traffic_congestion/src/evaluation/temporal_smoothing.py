"""
Temporal smoothing module for AGV zone congestion classifier.
Applies rolling window smoothing to raw per-timestep predictions.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATASET_PATH


def load_data():
    """Load the processed dataset with engineered features and true labels."""
    df = pd.read_csv(PROCESSED_DATASET_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def apply_temporal_smoothing(df, window_size=3):
    """
    Apply temporal smoothing using a rolling window of 3 timesteps.
    
    Decision rule (priority-based):
    - If ANY CRITICAL appears in window → CRITICAL
    - Else if WARNING appears ≥ 2 times → WARNING  
    - Else → SAFE
    
    Args:
        df: DataFrame with timestamp, zone_id, and predicted labels
        window_size: Size of the rolling window (default 3)
        
    Returns:
        DataFrame with smoothed predictions
    """
    df_sorted = df.sort_values(['zone_id', 'timestamp']).copy()
    
    # Add predicted states based on the labels
    label_to_state = {0: 'SAFE', 1: 'WARNING', 2: 'CRITICAL'}
    df_sorted['predicted_state'] = df_sorted['label'].map(label_to_state)
    
    # Convert states to numeric for easier processing
    state_to_num = {'SAFE': 0, 'WARNING': 1, 'CRITICAL': 2}
    df_sorted['state_num'] = df_sorted['predicted_state'].map(state_to_num)
    
    # Apply smoothing per zone
    smoothed_states = []
    
    for zone_id in df_sorted['zone_id'].unique():
        zone_data = df_sorted[df_sorted['zone_id'] == zone_id].reset_index(drop=True)
        
        # Initialize smoothed state column
        zone_smoothed_states = []
        
        for i in range(len(zone_data)):
            # Determine the start of the window
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            
            # Get the window of states
            window_states = zone_data.iloc[start_idx:end_idx]['state_num'].values
            
            # Apply decision rule
            if 2 in window_states:  # Any CRITICAL
                smoothed_state = 2  # CRITICAL
            elif sum(window_states == 1) >= 2:  # WARNING appears ≥ 2 times
                smoothed_state = 1  # WARNING
            else:
                smoothed_state = 0  # SAFE
            
            zone_smoothed_states.append(smoothed_state)
        
        # Map back to state names
        num_to_state = {0: 'SAFE', 1: 'WARNING', 2: 'CRITICAL'}
        zone_smoothed_state_names = [num_to_state[s] for s in zone_smoothed_states]
        
        # Add to the main dataframe
        zone_indices = df_sorted[df_sorted['zone_id'] == zone_id].index
        for idx, state in zip(zone_indices, zone_smoothed_state_names):
            smoothed_states.append((idx, state))
    
    # Create a mapping from index to smoothed state
    idx_to_smoothed = dict(smoothed_states)
    
    # Add the smoothed state column to the dataframe
    df_sorted['smoothed_state'] = df_sorted.index.map(idx_to_smoothed)
    
    return df_sorted


def save_smoothed_data(df, output_path):
    """Save the smoothed data to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Smoothed data saved to {output_path}")


def main():
    """Main function to run temporal smoothing."""
    print("Loading processed dataset...")
    df = load_data()
    
    print("Applying temporal smoothing...")
    df_smoothed = apply_temporal_smoothing(df)
    
    # Save the smoothed data
    output_path = Path(__file__).parent.parent / "data" / "processed" / "smh_dataset_temporal.csv"
    save_smoothed_data(df_smoothed, output_path)
    
    print("Temporal smoothing complete.")
    print(f"Smoothed dataset shape: {df_smoothed.shape}")
    print(f"State distribution in smoothed data:")
    print(df_smoothed['smoothed_state'].value_counts())


if __name__ == "__main__":
    main()