import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry

SAFE_TIME_GAP = 3.0
ZONE_X = 0.0   # conflict zone at x = 0


class ZoneManager(Node):

    def __init__(self):
        super().__init__('zone_manager')

        self.publisher_ = self.create_publisher(String, '/zone_permission', 10)

        self.sub_agv1 = self.create_subscription(
            Odometry, '/agv1/odom', self.agv1_callback, 10)

        self.sub_agv2 = self.create_subscription(
            Odometry, '/agv2/odom', self.agv2_callback, 10)

        self.timer = self.create_timer(1.0, self.evaluate_zone)

        self.agv1_data = None
        self.agv2_data = None

        self.get_logger().info("Zone Manager with dynamic odom started")

    def agv1_callback(self, msg):
        self.agv1_data = msg

    def agv2_callback(self, msg):
        self.agv2_data = msg

    def time_to_conflict(self, pos_x, speed):
        distance = abs(pos_x - ZONE_X)
        if speed <= 0:
            return float('inf')
        return distance / speed

    def evaluate_zone(self):
        if self.agv1_data is None or self.agv2_data is None:
            return

        p1 = self.agv1_data.pose.pose.position.x
        v1 = self.agv1_data.twist.twist.linear.x

        p2 = self.agv2_data.pose.pose.position.x
        v2 = self.agv2_data.twist.twist.linear.x

        t1 = self.time_to_conflict(p1, v1)
        t2 = self.time_to_conflict(p2, v2)

        msg = String()

        if abs(t2 - t1) >= SAFE_TIME_GAP:
            msg.data = "ALLOW"
        else:
            msg.data = "BLOCK"

        self.publisher_.publish(msg)

        self.get_logger().info(
            f"AGV1 ETA={t1:.2f}s | AGV2 ETA={t2:.2f}s → {msg.data}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ZoneManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
