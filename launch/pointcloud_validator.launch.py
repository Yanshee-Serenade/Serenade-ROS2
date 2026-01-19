"""
Launch file for the pointcloud validator node
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for pointcloud validator node"""
    
    pointcloud_node = Node(
        package='serenade_ros2',
        executable='pointcloud_validator_node',
        name='pointcloud_validator_node',
        output='screen',
    )
    
    return LaunchDescription([
        pointcloud_node,
    ])
