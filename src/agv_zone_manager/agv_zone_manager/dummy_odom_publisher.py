import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math


class DummyOdomPublisher(Node):

    def __init__(self):
        super().__init__('dummy_odom_publisher')

        self.pub_agv1 = self.create_publisher(Odometry, '/agv1/odom', 10)
        self.pub_agv2 = self.create_publisher(Odometry, '/agv2/odom', 10)

        self.timer = self.create_timer(1.0, self.publish_odom)

        self.pos1 = 10.0   # AGV1 starts far
        self.pos2 = 14.0   # AGV2 starts farther

        self.speed1 = 0.5
        self.speed2 = 0.5

        self.get_logger().info("Dummy Odom Publisher Started")

    def publish_odom(self):
        self.pos1 -= self.speed1
        self.pos2 -= self.speed2

        odom1 = Odometry()
        odom1.pose.pose.position.x = self.pos1
        odom1.twist.twist.linear.x = self.speed1

        odom2 = Odometry()
        odom2.pose.pose.position.x = self.pos2
        odom2.twist.twist.linear.x = self.speed2

        self.pub_agv1.publish(odom1)
        self.pub_agv2.publish(odom2)

        self.get_logger().info(
            f"AGV1 x={self.pos1:.2f}, AGV2 x={self.pos2:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = DummyOdomPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()