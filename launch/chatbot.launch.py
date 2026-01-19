"""
Launch file for the chatbot node
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for chatbot node"""
    
    chatbot_node = Node(
        package='serenade_ros2',
        executable='chatbot_node',
        name='chatbot_node',
        output='screen',
    )
    
    return LaunchDescription([
        chatbot_node,
    ])
