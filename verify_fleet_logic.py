import sys
import os
import unittest

# Add src to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src', 'fleet_manager')
sys.path.append(src_path)

# Mocking rclpy to allow importing nodes for syntax/logic checking if needed
# But for now we focus on the pure logic classes
from fleet_manager.path_planner import PathPlanner
from fleet_manager.task_allocator import TaskAllocator

class TestFleetManagerLogic(unittest.TestCase):
    def setUp(self):
        self.planner = PathPlanner()
        self.allocator = TaskAllocator()

    def test_path_planning_basic(self):
        print("\n[Testing] Path Planning (Basic)...")
        path = self.planner.find_path('A', 'D')
        print(f"Path A -> D: {path}")
        self.assertEqual(path, ['A', 'X', 'D'])
        
    def test_path_planning_congestion(self):
        print("\n[Testing] Path Planning (Congestion Awareness)...")
        # Base cost through X is 10+10 = 20.
        # If we set Traffic to CRITICAL, multiplier is 5.0. Cost = 20*5 = 100.
        # This graph is simple, so it might still be the only path.
        # Let's just verify the cost function logic.
        self.planner.update_traffic_state("CRITICAL")
        cost = self.planner.get_cost('A', 'X')
        print(f"Cost A->X (CRITICAL): {cost}")
        self.assertEqual(cost, 50.0) # 10 * 5
        
    def test_path_planning_blocked(self):
        print("\n[Testing] Path Planning (Zone Blocked)...")
        self.planner.update_zone_status('X', 'BLOCK')
        path = self.planner.find_path('A', 'D')
        print(f"Path A -> D (X Blocked): {path}")
        self.assertIsNone(path) # Should be blocked

    def test_task_allocation(self):
        print("\n[Testing] Task Allocation...")
        self.allocator.add_task('T1', 'A', 'D')
        assignments = self.allocator.assign_tasks()
        print(f"Assignments: {assignments}")
        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0]['agv_id'], 'agv1')
        self.assertEqual(assignments[0]['task']['id'], 'T1')

if __name__ == '__main__':
    unittest.main()
