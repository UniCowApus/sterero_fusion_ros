from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='stereo_fusion_node',
            executable='stereo_fusion_node',
            name='stereo_fusion',
            output='screen'
        )
    ])
