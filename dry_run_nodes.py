import sys
import os
import unittest
from unittest.mock import MagicMock

# Mock module 'rclpy' and 'std_msgs' BEFORE importing our nodes
sys.modules['rclpy'] = MagicMock()
sys.modules['rclpy.node'] = MagicMock()
sys.modules['std_msgs'] = MagicMock()
sys.modules['std_msgs.msg'] = MagicMock()
sys.modules['geometry_msgs'] = MagicMock()
sys.modules['geometry_msgs.msg'] = MagicMock()

# Define Mock Node class because our nodes inherit from it
class MockNode:
    def __init__(self, name):
        self.name = name
    def create_subscription(self, *args): pass
    def create_publisher(self, *args): pass
    def create_timer(self, *args): pass
    def get_logger(self): return MagicMock()

sys.modules['rclpy.node'].Node = MockNode

# Add src path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src', 'fleet_manager')
sys.path.append(src_path)

# Import our Nodes
from fleet_manager.fleet_manager_node import FleetManagerNode
from fleet_manager.traffic_prediction_node import TrafficPredictionNode

class TestNodeInstantiation(unittest.TestCase):
    def test_fleet_manager_init(self):
        print("\n[Testing] FleetManagerNode Instantiation...")
        node = FleetManagerNode()
        self.assertIsNotNone(node)
        print("FleetManagerNode initialized successfully.")

    def test_traffic_prediction_init(self):
        print("\n[Testing] TrafficPredictionNode Instantiation...")
        node = TrafficPredictionNode()
        self.assertIsNotNone(node)
        print("TrafficPredictionNode initialized successfully.")

if __name__ == '__main__':
    unittest.main()
