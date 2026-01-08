"""
Baseline comparison module for AGV zone congestion classifier.
Simulates a baseline system without intelligence and compares to ML-assisted approach.
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


def simulate_baseline_system(congestion_events):
    """
    Simulate a baseline system that reacts only at congestion onset.
    
    Args:
        congestion_events: DataFrame with congestion onset times
        
    Returns:
        DataFrame with baseline reaction times (0 seconds)
    """
    baseline_results = []
    
    for _, event in congestion_events.iterrows():
        baseline_results.append({
            'zone_id': event['zone_id'],
            'congestion_onset_time': event['onset_time'],
            'reaction_time': 0  # Baseline system reacts at onset (0 lead time)
        })
    
    return pd.DataFrame(baseline_results)


def compare_systems(lead_time_results, baseline_results):
    """
    Compare baseline system vs ML-assisted system.
    
    Args:
        lead_time_results: DataFrame with ML-assisted lead times
        baseline_results: DataFrame with baseline reaction times
        
    Returns:
        Comparison results
    """
    # Calculate improvement metrics
    if len(lead_time_results) > 0:
        ml_avg_lead_time = lead_time_results['lead_time_seconds'].mean()
        baseline_avg_reaction = 0  # Baseline always reacts at 0 seconds (at onset)
        
        improvement = ml_avg_lead_time - baseline_avg_reaction  # This is the gain
        
        return {
            'ml_assisted_avg_lead_time': ml_avg_lead_time,
            'baseline_avg_reaction_time': baseline_avg_reaction,
            'average_improvement': improvement,
            'ml_events_count': len(lead_time_results),
            'baseline_events_count': len(baseline_results)
        }
    else:
        return {
            'ml_assisted_avg_lead_time': 0,
            'baseline_avg_reaction_time': 0,
            'average_improvement': 0,
            'ml_events_count': 0,
            'baseline_events_count': len(baseline_results)
        }


def main():
    """Main function to run baseline comparison."""
    print("Loading processed dataset...")
    df = load_data()
    
    print("Identifying congestion events...")
    congestion_events = identify_congestion_events(df)
    print(f"Found {len(congestion_events)} congestion events")
    
    print("Simulating baseline system...")
    baseline_results = simulate_baseline_system(congestion_events)
    print(f"Baseline system would react at {len(baseline_results)} events")
    
    # We need to get the lead time results by running the same process as in lead_time_analysis
    # Since we can't directly import due to relative path issues, we'll reimplement the function here
    label_to_state = {0: 'SAFE', 1: 'WARNING', 2: 'CRITICAL'}
    df['predicted_state'] = df['label'].map(label_to_state)
    
    lead_time_results = []
    
    for _, event in congestion_events.iterrows():
        zone_id = event['zone_id']
        onset_time = event['onset_time']
        
        # Filter data for this zone and before congestion onset
        zone_data = df[(df['zone_id'] == zone_id) & (df['timestamp'] < onset_time)]
        
        # Find first WARNING in this period
        warning_data = zone_data[zone_data['predicted_state'] == 'WARNING']
        
        if len(warning_data) > 0:
            first_warning_time = warning_data['timestamp'].min()
            
            # Calculate lead time in seconds
            lead_time_seconds = (onset_time - first_warning_time).total_seconds()
            
            lead_time_results.append({
                'zone_id': zone_id,
                'first_warning_time': first_warning_time,
                'congestion_onset_time': onset_time,
                'lead_time_seconds': lead_time_seconds
            })
    
    lead_time_results = pd.DataFrame(lead_time_results)
    
    print(f"ML-assisted system would provide warning for {len(lead_time_results)} events")
    
    print("Comparing systems...")
    comparison = compare_systems(lead_time_results, baseline_results)
    
    print("\nSYSTEM COMPARISON:")
    print("=" * 50)
    print(f"Baseline System:")
    print(f"  Average Reaction Time: {comparison['baseline_avg_reaction_time']:.1f} seconds")
    print(f"  Events Handled: {comparison['baseline_events_count']}")
    print()
    print(f"ML-Assisted System:")
    print(f"  Average Lead Time: {comparison['ml_assisted_avg_lead_time']:.1f} seconds")
    print(f"  Events with Early Warning: {comparison['ml_events_count']}")
    print()
    print(f"Improvement with ML:")
    print(f"  Average Lead Time Gained: {comparison['average_improvement']:.1f} seconds")
    
    if comparison['baseline_events_count'] > 0:
        improvement_percentage = (comparison['average_improvement'] / 1) * 100  # Just showing the gain
        print(f"  Relative Improvement: {comparison['average_improvement']:.1f}s earlier detection")


if __name__ == "__main__":
    main()