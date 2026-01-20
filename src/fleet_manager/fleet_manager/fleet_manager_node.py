import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Odometry
import math

# Import internal modules
from fleet_manager.path_planner import PathPlanner
from fleet_manager.task_allocator import TaskAllocator

class FleetManagerNode(Node):
    def __init__(self):
        super().__init__('fleet_manager')
        
        # Subsystems
        self.planner = PathPlanner()
        self.allocator = TaskAllocator()
        
        # Subscribers
        self.create_subscription(String, '/zone_permission', self.zone_callback, 10)
        self.create_subscription(String, '/traffic_congestion', self.traffic_callback, 10)
        self.create_subscription(String, '/erp_tasks', self.erp_task_callback, 10)
        
        self.create_subscription(Odometry, '/agv1/odom', self.agv1_odom_callback, 10)
        self.create_subscription(Odometry, '/agv2/odom', self.agv2_odom_callback, 10)
        
        # Publishers
        self.agv1_pub = self.create_publisher(Twist, '/agv1/cmd_vel', 10)
        self.agv2_pub = self.create_publisher(Twist, '/agv2/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/fleet_status', 10)
        
        # State
        self.zone_permission = "ALLOW"
        self.agv_positions = {'agv1': (0,0), 'agv2': (0,0)} # x, y
        
        # Control Loop
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("Fleet Manager Node Started")

    def zone_callback(self, msg):
        self.zone_permission = msg.data
        self.planner.update_zone_status('X', msg.data) # Assuming X is the critical zone

    def traffic_callback(self, msg):
        self.planner.update_traffic_state(msg.data)
        self.get_logger().info(f"Updated Traffic State: {msg.data}")

    def erp_task_callback(self, msg):
        # Format: "TaskID:Source:Destination" e.g., "T1:A:D"
        try:
            parts = msg.data.split(':')
            if len(parts) == 3:
                t_id, src, dst = parts
                self.allocator.add_task(t_id, src, dst)
                self.get_logger().info(f"Received Task {t_id}: {src} -> {dst}")
        except Exception as e:
            self.get_logger().error(f"Invalid Task Format: {msg.data}")

    def agv1_odom_callback(self, msg):
        self.agv_positions['agv1'] = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.check_node_arrival('agv1')

    def agv2_odom_callback(self, msg):
        self.agv_positions['agv2'] = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.check_node_arrival('agv2')

    def check_node_arrival(self, agv_id):
        # Update Task Allocator status if AGV reaches destination
        pass # Simplified for now

    def control_loop(self):
        # 1. Assign Tasks
        new_assignments = self.allocator.assign_tasks()
        for assignment in new_assignments:
            self.get_logger().info(f"Assigned {assignment['task']['id']} to {assignment['agv_id']}")
        
        # 2. Control AGVs
        for agv_id in ['agv1', 'agv2']:
            self.move_agv(agv_id)

    def move_agv(self, agv_id):
        current_task_id = self.allocator.agves[agv_id]['current_task']
        if not current_task_id:
            return # Idle

        # Get goal from task
        # Simplified: We just know Source and Destination. 
        # Creating a path requires finding current location's nearest node.
        # For simplicity, we assume generic movement towards goal.
        
        # Check Safety
        if self.zone_permission == "BLOCK":
            # If AGV is near intersection (Zone X), stop it.
            # Zone X is at (0,0).
            x, y = self.agv_positions[agv_id]
            dist_to_x = math.sqrt(x**2 + y**2)
            
            if dist_to_x < 5.0 and dist_to_x > 1.0: 
                # Approaching conflict
                self.send_stop(agv_id)
                return

        # Move Logic (Simple P-Controller towards destination)
        # TODO: Get full path from planner
        # For this demo, just move
        self.send_velocity(agv_id, 0.5, 0.0) # Move forward

    def send_stop(self, agv_id):
        twist = Twist()
        if agv_id == 'agv1':
            self.agv1_pub.publish(twist)
        else:
            self.agv2_pub.publish(twist)

    def send_velocity(self, agv_id, linear, angular):
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        if agv_id == 'agv1':
            self.agv1_pub.publish(twist)
        else:
            self.agv2_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = FleetManagerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
