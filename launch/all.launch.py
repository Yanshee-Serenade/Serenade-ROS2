"""
Combined launch file for all ROS2 nodes
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for all nodes"""
    
    chatbot_node = Node(
        package='serenade_ros2',
        executable='chatbot_node',
        name='chatbot_node',
        output='screen',
    )
    
    vlm_server_node = Node(
        package='serenade_ros2',
        executable='vlm_server_node',
        name='vlm_server_node',
        output='screen',
    )
    
    walker_node = Node(
        package='serenade_ros2',
        executable='walker_node',
        name='walker_node',
        output='screen',
    )
    
    return LaunchDescription([
        chatbot_node,
        vlm_server_node,
        walker_node,
    ])
