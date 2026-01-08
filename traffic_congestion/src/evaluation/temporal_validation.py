"""
Temporal validation module for AGV zone congestion classifier.
Verifies that WARNING states appear before CRITICAL states in most congestion events.
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


def identify_congestion_events(df, congestion_threshold=0.7, consecutive_steps=2):
    """
    Identify actual congestion onset events using avg_zone_speed <= 0.7 for 2 consecutive time steps.
    
    Args:
        df: DataFrame with timestamp, zone_id, and avg_zone_speed
        congestion_threshold: Speed threshold for congestion (default 0.7 m/s)
        consecutive_steps: Number of consecutive steps to consider as congestion onset
        
    Returns:
        DataFrame with congestion events per zone
    """
    congestion_events = []
    
    for zone_id in df['zone_id'].unique():
        zone_data = df[df['zone_id'] == zone_id].sort_values('timestamp')
        
        # Find where speed drops below threshold
        speed_low = zone_data['avg_zone_speed'] <= congestion_threshold
        
        # Find consecutive sequences of low speed using a boolean mask approach
        # Create a mask where True indicates consecutive low speeds
        if len(speed_low) >= consecutive_steps:
            # Use a sliding window approach
            consecutive_mask = []
            for i in range(len(speed_low) - consecutive_steps + 1):
                # Check if consecutive_steps values starting from i are all True
                if all(speed_low.iloc[i:i+consecutive_steps]):
                    # This index marks the beginning of consecutive low speeds
                    original_idx = speed_low.index[i]
                    onset_time = zone_data.loc[original_idx, 'timestamp']
                    congestion_events.append({
                        'zone_id': zone_id,
                        'onset_time': onset_time,
                        'onset_index': original_idx
                    })
                    break  # Only take the first congestion event per zone for now
    
    return pd.DataFrame(congestion_events)


def analyze_state_transitions(df, congestion_events):
    """
    Analyze state transitions over time per zone to verify WARNING appears before CRITICAL.
    
    Args:
        df: DataFrame with timestamp, zone_id, and predicted labels
        congestion_events: DataFrame with congestion onset times
        
    Returns:
        Analysis results
    """
    results = []
    
    for zone_id in df['zone_id'].unique():
        zone_data = df[df['zone_id'] == zone_id].sort_values('timestamp').copy()
        
        # Add predicted states based on the labels
        label_to_state = {0: 'SAFE', 1: 'WARNING', 2: 'CRITICAL'}
        zone_data['predicted_state'] = zone_data['label'].map(label_to_state)
        
        # Get congestion onset for this zone
        zone_congestion = congestion_events[congestion_events['zone_id'] == zone_id]
        
        if len(zone_congestion) > 0:
            onset_time = zone_congestion.iloc[0]['onset_time']
            
            # Find all states before the congestion onset
            pre_onset = zone_data[zone_data['timestamp'] < onset_time]
            post_onset = zone_data[zone_data['timestamp'] >= onset_time]
            
            # Check if WARNING appeared before CRITICAL
            pre_onset_states = pre_onset['predicted_state'].unique()
            post_onset_states = post_onset['predicted_state'].unique()
            
            # Find first occurrence of each state before onset
            first_warning = pre_onset[pre_onset['predicted_state'] == 'WARNING']['timestamp'].min()
            first_critical = pre_onset[pre_onset['predicted_state'] == 'CRITICAL']['timestamp'].min()
            
            # Check if WARNING appeared before any CRITICAL
            warning_before_critical = False
            if pd.notna(first_warning) and pd.notna(first_critical):
                warning_before_critical = first_warning < first_critical
            elif pd.notna(first_warning) and pd.isna(first_critical):
                warning_before_critical = True  # WARNING appeared but no CRITICAL before onset
            
            results.append({
                'zone_id': zone_id,
                'onset_time': onset_time,
                'first_warning_time': first_warning,
                'first_critical_time': first_critical,
                'warning_before_critical': warning_before_critical,
                'pre_onset_states': list(pre_onset_states),
                'post_onset_states': list(post_onset_states)
            })
    
    return results


def print_state_transitions(transition_results):
    """Print state transition sequences with timestamps."""
    print("State Transition Analysis Results:")
    print("=" * 50)
    
    for result in transition_results:
        print(f"Zone: {result['zone_id']}")
        print(f"  Congestion Onset: {result['onset_time']}")
        print(f"  First WARNING: {result['first_warning_time']}")
        print(f"  First CRITICAL: {result['first_critical_time']}")
        print(f"  WARNING Before CRITICAL: {result['warning_before_critical']}")
        print(f"  Pre-onset States: {result['pre_onset_states']}")
        print(f"  Post-onset States: {result['post_onset_states']}")
        print()


def main():
    """Main function to run temporal validation."""
    print("Loading processed dataset...")
    df = load_data()
    
    print("Identifying congestion events...")
    congestion_events = identify_congestion_events(df)
    print(f"Found {len(congestion_events)} congestion events")
    
    print("Analyzing state transitions...")
    transition_results = analyze_state_transitions(df, congestion_events)
    
    print_state_transitions(transition_results)
    
    # Count how many zones had WARNING before CRITICAL
    warning_before_critical_count = sum(1 for r in transition_results if r['warning_before_critical'])
    total_zones = len(transition_results)
    
    if total_zones > 0:
        print(f"SUMMARY:")
        print(f"  Zones with WARNING before CRITICAL: {warning_before_critical_count}/{total_zones}")
        print(f"  Percentage: {warning_before_critical_count/total_zones*100:.1f}%")
    else:
        print("No congestion events found to analyze.")


if __name__ == "__main__":
    main()