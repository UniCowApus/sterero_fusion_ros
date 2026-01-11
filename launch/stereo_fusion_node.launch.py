from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    log_level = LaunchConfiguration('log_level')

    return LaunchDescription([
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Logging level'
        ),

        Node(
            package='stereo_fusion_node',
            executable='stereo_fusion_node',
            name='stereo_fusion',
            output='screen',
            arguments=['--ros-args', '--log-level', log_level],
            respawn=True,
            respawn_delay=2.0,
        )
    ])

