"""
Visualization module for AGV zone congestion classifier.
Generates timeline plot for smoothed states and lead-time comparison.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATASET_PATH, ARTIFACTS_DIR


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


def plot_zone_state_timeline_smoothed(df, zone_id='Z1', save_path=None):
    """
    Plot timeline for a single zone showing smoothed predicted states over time.
    
    Args:
        df: DataFrame with timestamp, zone_id, and smoothed predicted labels
        zone_id: Zone to plot
        save_path: Path to save the plot
    """
    # Filter for specific zone
    zone_data = df[df['zone_id'] == zone_id].sort_values('timestamp')
    
    # Convert states to numeric for plotting
    state_to_num = {'SAFE': 0, 'WARNING': 1, 'CRITICAL': 2}
    zone_data['state_numeric'] = zone_data['smoothed_state'].map(state_to_num)
    
    # Identify congestion onset for this zone
    congestion_events = identify_congestion_events(zone_data)
    
    # Create the plot
    plt.figure(figsize=(14, 6))
    
    # Plot the state over time
    plt.plot(zone_data['timestamp'], zone_data['state_numeric'], 
             marker='o', linestyle='-', linewidth=2, markersize=4)
    
    # Mark congestion onset times
    for _, event in congestion_events.iterrows():
        plt.axvline(x=event['onset_time'], color='red', linestyle='--', linewidth=2,
                   label='Actual Congestion Onset' if event.name == congestion_events.index[0] else "")
    
    # Customize the plot
    plt.title(f'Zone {zone_id} - Temporally Smoothed Predicted State Timeline', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Predicted State', fontsize=12)
    plt.yticks([0, 1, 2], ['SAFE', 'WARNING', 'CRITICAL'])
    plt.grid(True, alpha=0.3)
    
    # Add legend if there are congestion events
    if len(congestion_events) > 0:
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Smoothed timeline plot saved to {save_path}")
    
    plt.show()


def plot_lead_time_comparison(lead_time_results, save_path=None):
    """
    Plot lead-time comparison between baseline and ML-assisted systems.
    
    Args:
        lead_time_results: DataFrame with ML-assisted lead times
        save_path: Path to save the plot
    """
    if len(lead_time_results) == 0:
        print("No lead time results to plot.")
        return
    
    # Prepare data for plotting
    n_events = len(lead_time_results)
    event_indices = range(n_events)
    
    # Extract lead times
    lead_times = lead_time_results['lead_time_seconds'].values
    
    # Baseline is always 0 (reacts at congestion onset)
    baseline_times = np.zeros(n_events)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    x = np.arange(n_events)  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - width/2, baseline_times, width, label='Baseline (0s)', color='lightcoral')
    bars2 = ax.bar(x + width/2, lead_times, width, label='ML-assisted (lead time)', color='lightblue')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Congestion Event Index')
    ax.set_ylabel('Reaction Time (seconds)')
    ax.set_title('Lead Time Comparison: Baseline vs ML-Assisted System')
    ax.set_xticks(x)
    ax.legend()

    # Add value labels on bars
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                    f'{value:.1f}s',
                    ha='center', va='bottom', fontsize=9)

    add_value_labels(bars1, baseline_times)
    add_value_labels(bars2, lead_times)

    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Lead time comparison plot saved to {save_path}")
    
    plt.show()


def main():
    """Main function to generate visualizations."""
    print("Loading smoothed dataset...")
    df = load_smoothed_data()
    
    if df is None:
        return
    
    print("Creating plots directory...")
    plots_dir = ARTIFACTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Timeline for a single zone (Z1 as specified)
    timeline_plot_path = plots_dir / "zone_state_timeline_smoothed.png"
    print(f"Generating smoothed timeline plot for zone Z1...")
    plot_zone_state_timeline_smoothed(df, zone_id='Z1', save_path=timeline_plot_path)
    
    # Identify congestion events and compute lead times for comparison plot
    print("Identifying congestion events...")
    congestion_events = identify_congestion_events(df)
    
    print("Computing lead times...")
    lead_time_results = find_first_warning_before_congestion(df, congestion_events)
    
    # Plot 2: Lead-time comparison
    comparison_plot_path = plots_dir / "lead_time_comparison.png"
    print("Generating lead-time comparison plot...")
    plot_lead_time_comparison(lead_time_results, save_path=comparison_plot_path)
    
    print(f"\nVisualizations generated and saved to {plots_dir}")
    
    # Print summary
    if len(lead_time_results) > 0:
        valid_warnings = sum(1 for lt in lead_time_results['lead_time_seconds'] if lt > 0)
        avg_lead_time = lead_time_results[lead_time_results['lead_time_seconds'] > 0]['lead_time_seconds'].mean() if valid_warnings > 0 else 0
        
        print(f"\nSUMMARY:")
        print(f"  Number of congestion events: {len(lead_time_results)}")
        print(f"  Events with WARNING before congestion: {valid_warnings}")
        if len(lead_time_results) > 0:
            warning_percentage = (valid_warnings / len(lead_time_results)) * 100
            print(f"  % where WARNING preceded CRITICAL: {warning_percentage:.1f}%")
        print(f"  Average lead time gained: {avg_lead_time:.1f} seconds")


if __name__ == "__main__":
    main()