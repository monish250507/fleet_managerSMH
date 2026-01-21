import sys
import os
import time

# --- SETUP PATHS ---
# We need to add:
# 1. fleet_manager src
# 2. traffic_congestion src

current_dir = os.path.dirname(os.path.abspath(__file__))
fleet_src_path = os.path.join(current_dir, 'src', 'fleet_manager')
traffic_src_path = os.path.join(current_dir, 'traffic_congestion', 'src')

sys.path.append(fleet_src_path)
sys.path.append(traffic_src_path)

print(f"[INFO] Added paths:\n  - {fleet_src_path}\n  - {traffic_src_path}")

# --- IMPORTS ---
try:
    from fleet_manager.path_planner import PathPlanner
    from fleet_manager.task_allocator import TaskAllocator
    from models.inference import ZoneCongestionPredictor
    print("[SUCCESS] All modules imported successfully.")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("Ensure you are running this script from the workspace root (e.g., deadlock-logic folder).")
    sys.exit(1)

# --- DEMO CLASS ---
class IntegratedFleetDemo:
    def __init__(self):
        print("\n--- INITIALIZING INTEGRATED FLEET MANAGER DEMO ---")
        
        # 1. Initialize ML Predictor
        try:
            print("Loading ML Model...")
            self.predictor = ZoneCongestionPredictor()
            print("[OK] ML Model Loaded.")
        except Exception as e:
            print(f"[FAIL] Could not load ML model: {e}")
            sys.exit(1)

        # 2. Initialize Fleet Components
        self.planner = PathPlanner()
        self.allocator = TaskAllocator()
        print("[OK] Path Planner & Task Allocator Initialized.")

    def run_scenario(self, scenario_name, data_snapshot):
        print(f"\n\n>>> SCENARIO: {scenario_name} <<<")
        print("-" * 40)
        
        # 1. Predict Traffic Congestion
        print(f"Input Data: {data_snapshot}")
        prediction = self.predictor.predict(data_snapshot)
        state = prediction['state']
        print(f"ML PREDICTION: [{state}] (Dominant Signals: {prediction['dominant_signals']})")
        
        # 2. Update Fleet Manager Logic
        self.planner.update_traffic_state(state)
        
        # 3. Path Planning Check
        # Let's say we want to go from A to D, passing through X
        print("\n[Path Planning Request: A -> D]")
        
        # Check cost of critical segment (A->X or through X)
        # Note: In our simple graph, X connects everything.
        cost_ax = self.planner.get_cost('A', 'X')
        print(f"Cost A->X: {cost_ax}")
        
        path = self.planner.find_path('A', 'D')
        if path:
            print(f"Calculated Path: {path}")
        else:
            print("Calculated Path: NO PATH FOUND (Blocked)")

        return state, path

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    demo = IntegratedFleetDemo()

    # Scenario 1: SAFE (Low traffic)
    safe_data = {
        'task_arrival_rate': 2.0,
        'zone_agv_count': 1,
        'avg_zone_speed': 1.2,
        'path_overlap_score': 0.1,
        'zone_density': 0.2,
        'avg_zone_wait_time': 2.0
    }
    demo.run_scenario("Normal Operations (Low Traffic)", safe_data)

    # Scenario 2: WARNING (Medium traffic)
    warning_data = {
        'task_arrival_rate': 6.0,
        'zone_agv_count': 3,
        'avg_zone_speed': 0.8,
        'path_overlap_score': 0.4,
        'zone_density': 0.5,
        'avg_zone_wait_time': 15.0
    }
    demo.run_scenario("Busy Shift (Medium Traffic)", warning_data)

    # Scenario 3: CRITICAL (Heavy congestion)
    critical_data = {
        'task_arrival_rate': 9.0,
        'zone_agv_count': 6,
        'avg_zone_speed': 0.3,
        'path_overlap_score': 0.7,
        'zone_density': 0.9,
        'avg_zone_wait_time': 35.0
    }
    demo.run_scenario("Gridlock (Heavy Traffic)", critical_data)
    
    print("\n\n[DEMO COMPLETE] The Fleet Manager successfully adjusted path costs based on live ML predictions.")
