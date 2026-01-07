import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import paho.mqtt.client as mqtt


class RosMqttBridge(Node):

    def __init__(self):
        super().__init__('ros_mqtt_bridge')

        # MQTT client
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect("localhost", 1883, 60)
        self.mqtt_client.loop_start()

        # ROS subscription
        self.subscription = self.create_subscription(
            String,
            '/zone_permission',
            self.zone_permission_callback,
            10
        )

        self.get_logger().info("ROS ↔ MQTT Bridge Node Started")

    def zone_permission_callback(self, msg):
        self.mqtt_client.publish('/zone/permission', msg.data)
        self.get_logger().info(f"Forwarded to MQTT: {msg.data}")


def main(args=None):
    rclpy.init(args=args)
    node = RosMqttBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
