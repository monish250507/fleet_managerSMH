"""
Congestion onset detection module for AGV zone congestion classifier.
Defines actual congestion onset as avg_zone_speed <= 0.7 for 2 consecutive timesteps.
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


def main():
    """Main function to identify congestion onset events."""
    print("Loading processed dataset...")
    df = load_data()
    
    print("Identifying congestion events...")
    congestion_events = identify_congestion_events(df)
    
    print(f"Found {len(congestion_events)} congestion events")
    
    if len(congestion_events) > 0:
        print("\nCongestion Events:")
        print("=" * 50)
        for _, event in congestion_events.iterrows():
            print(f"Zone: {event['zone_id']}")
            print(f"  Onset Time: {event['onset_time']}")
            print(f"  Index: {event['onset_index']}")
            print()
    
    # Save congestion events for later use
    output_path = Path(__file__).parent.parent / "data" / "processed" / "congestion_onset_events.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    congestion_events.to_csv(output_path, index=False)
    print(f"Congestion events saved to {output_path}")


if __name__ == "__main__":
    main()