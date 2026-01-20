from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='fleet_manager',
            executable='traffic_predictor',
            name='traffic_prediction_node',
            output='screen'
        ),
        Node(
            package='fleet_manager',
            executable='fleet_manager',
            name='fleet_manager_node',
            output='screen'
        ),
    ])
