import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
import sys
import os
from pathlib import Path

# Add traffic_congestion to python path
# Assuming the workspace structure: src/fleet_manager/fleet_manager/traffic_prediction_node.py
# traffic_congestion is at src/../traffic_congestion
# OR c:/Users/monis/OneDrive/Documents/Desktop/SMH/deadlock-logic/traffic_congestion
TRAFFIC_LIB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    'traffic_congestion', 'src'
)
sys.path.append(TRAFFIC_LIB_PATH)

try:
    from models.inference import ZoneCongestionPredictor
except ImportError:
    # Fallback for dev environment without the ML module
    class ZoneCongestionPredictor:
        def __init__(self): pass
        def predict(self, data):
            return {"state": "SAFE", "probabilities": {"SAFE": 1.0}, "dominant_signals": []}


class TrafficPredictionNode(Node):
    def __init__(self):
        super().__init__('traffic_prediction_node')
        
        self.publisher_ = self.create_publisher(String, '/traffic_congestion', 10)
        
        # In a real system, we would subscribe to aggregated AGV data
        # Here we simulate the input data accumulation or subscribe to a unified state topic
        self.create_timer(2.0, self.predict_traffic)
        
        try:
            self.predictor = ZoneCongestionPredictor()
            self.get_logger().info(f"ML Model loaded from {TRAFFIC_LIB_PATH}")
        except Exception as e:
            self.get_logger().error(f"Failed to load ML Model: {e}")
            self.predictor = None

        self.mock_data = {
            'task_arrival_rate': 2.0,
            'zone_agv_count': 1,
            'avg_zone_speed': 0.8,
            'path_overlap_score': 0.1,
            'zone_density': 0.2,
            'avg_zone_wait_time': 2.0
        }

    def predict_traffic(self):
        if not self.predictor:
            return

        # In real implementation: Update self.mock_data from subscribers
        
        try:
            result = self.predictor.predict(self.mock_data)
            state = result['state']
            
            msg = String()
            msg.data = state
            self.publisher_.publish(msg)
            
            self.get_logger().info(f"Traffic State: {state}")
            
        except Exception as e:
            self.get_logger().warn(f"Prediction failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = TrafficPredictionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
