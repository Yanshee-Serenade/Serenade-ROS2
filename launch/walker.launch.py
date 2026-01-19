"""
Launch file for the walker node
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for walker node"""
    
    walker_node = Node(
        package='serenade_ros2',
        executable='walker_node',
        name='walker_node',
        output='screen',
    )
    
    return LaunchDescription([
        walker_node,
    ])
