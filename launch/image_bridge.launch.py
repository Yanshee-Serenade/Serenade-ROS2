"""
Launch file for the VLM server node
"""

from launch import LaunchDescription
from launch_ros.actions import ExecuteProcess


def generate_launch_description():
    """Generate launch description for VLM server node"""
    
    vlm_server_node = ExecuteProcess(
        cmd=['python3', '-O', '-m', 'serenade_ros2.image_bridge_node'],
        name='image_bridge_node',
        output='screen',
    )
    
    return LaunchDescription([
        vlm_server_node,
    ])
