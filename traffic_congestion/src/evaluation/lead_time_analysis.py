"""
Lead time analysis module for AGV zone congestion classifier.
Computes lead time between smoothed WARNING predictions and actual congestion onset.
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


def load_smoothed_data():
    """Load the temporally smoothed dataset."""
    try:
        df = pd.read_csv(Path(__file__).parent.parent / "data" / "processed" / "smh_dataset_temporal.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        print("Smoothed data not found. Please run temporal_smoothing.py first.")
        return None


def identify_congestion_events(df, congestion_threshold=0.7, consecutive_steps=2):
    """
    Identify actual congestion onset events using avg_zone_speed <= 0.7 for 2 consecutive timesteps.
    
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
        if len(speed_low) >= consecutive_steps:
            # Use a sliding window approach
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


def find_first_warning_before_congestion(df, congestion_events):
    """
    Find the first WARNING state before each congestion event (strictly before onset).
    
    Args:
        df: DataFrame with timestamp, zone_id, and predicted labels
        congestion_events: DataFrame with congestion onset times
        
    Returns:
        DataFrame with warning times and lead times
    """
    # Add predicted states based on the labels
    label_to_state = {0: 'SAFE', 1: 'WARNING', 2: 'CRITICAL'}
    df['predicted_state'] = df['label'].map(label_to_state)
    
    lead_time_results = []
    
    for _, event in congestion_events.iterrows():
        zone_id = event['zone_id']
        onset_time = event['onset_time']
        
        # Filter data for this zone and BEFORE (strictly) congestion onset
        zone_data = df[(df['zone_id'] == zone_id) & (df['timestamp'] < onset_time)]
        
        # Find WARNING in this period
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
        else:
            # No WARNING before onset, mark lead time as 0
            lead_time_results.append({
                'zone_id': zone_id,
                'first_warning_time': None,
                'congestion_onset_time': onset_time,
                'lead_time_seconds': 0
            })
    
    return pd.DataFrame(lead_time_results)


def compute_lead_time_metrics(lead_time_results):
    """
    Compute aggregate lead time metrics.
    
    Args:
        lead_time_results: DataFrame with warning times and lead times
        
    Returns:
        Dictionary with metrics
    """
    if len(lead_time_results) == 0:
        return {
            'average_lead_time': 0,
            'min_lead_time': 0,
            'max_lead_time': 0,
            'valid_events_count': 0,
            'warning_before_congestion_count': 0
        }
    
    # Count events where WARNING preceded congestion (lead_time > 0)
    warning_before_count = sum(1 for lt in lead_time_results['lead_time_seconds'] if lt > 0)
    
    lead_times = lead_time_results['lead_time_seconds'].values
    
    # Calculate metrics excluding zero lead times for avg, min, max when there are valid warnings
    positive_lead_times = [lt for lt in lead_times if lt > 0]
    
    metrics = {
        'average_lead_time': np.mean(positive_lead_times) if positive_lead_times else 0,
        'min_lead_time': np.min(positive_lead_times) if positive_lead_times else 0,
        'max_lead_time': np.max(positive_lead_times) if positive_lead_times else 0,
        'valid_events_count': len(lead_times),
        'warning_before_congestion_count': warning_before_count
    }
    
    return metrics


def print_summary(metrics):
    """Print the summary in the required format."""
    if metrics['valid_events_count'] > 0:
        warning_percentage = (metrics['warning_before_congestion_count'] / metrics['valid_events_count']) * 100
        print()
        print("=" * 50)
        print("EVALUATION SUMMARY:")
        print(f"WARNING precedes congestion in {warning_percentage:.1f}% of events")
        print(f"Average lead time gained: {metrics['average_lead_time']:.1f} seconds")
        print("Early warning enables preventive control action")
        print("=" * 50)
    else:
        print("No congestion events found for analysis.")


def main():
    """Main function to run lead time analysis."""
    print("Loading smoothed dataset...")
    df = load_smoothed_data()
    
    if df is None:
        return
    
    print("Identifying congestion events...")
    congestion_events = identify_congestion_events(df)
    print(f"Found {len(congestion_events)} congestion events")
    
    print("Finding first WARNING before each congestion event...")
    lead_time_results = find_first_warning_before_congestion(df, congestion_events)
    
    print(f"Analyzed {len(lead_time_results)} events")
    
    if len(lead_time_results) > 0:
        print("\nLead Time Results:")
        print("=" * 50)
        for _, result in lead_time_results.iterrows():
            print(f"Zone: {result['zone_id']}")
            print(f"  First WARNING: {result['first_warning_time']}")
            print(f"  Congestion Onset: {result['congestion_onset_time']}")
            print(f"  Lead Time: {result['lead_time_seconds']:.1f} seconds")
            print()
    
    print("Computing aggregate metrics...")
    metrics = compute_lead_time_metrics(lead_time_results)
    
    print("\nAGGREGATE METRICS:")
    print("=" * 50)
    print(f"Number of congestion events: {metrics['valid_events_count']}")
    print(f"% events where WARNING preceded congestion: {(metrics['warning_before_congestion_count']/metrics['valid_events_count']*100) if metrics['valid_events_count'] > 0 else 0:.1f}%")
    print(f"Average lead time: {metrics['average_lead_time']:.1f} seconds")
    print(f"Minimum lead time: {metrics['min_lead_time']:.1f} seconds")
    print(f"Maximum lead time: {metrics['max_lead_time']:.1f} seconds")
    
    print_summary(metrics)


if __name__ == "__main__":
    main()