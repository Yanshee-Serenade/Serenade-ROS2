"""
Launch file for the VLM server node
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for VLM server node"""
    
    # Declare launch arguments
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/image_slow',
        description='The image topic to subscribe to'
    )
    
    model_name_arg = DeclareLaunchArgument(
        'model_name',
        default_value='Qwen/Qwen3-VL-8B-Instruct',
        description='The VLM model name to use'
    )
    
    max_new_tokens_arg = DeclareLaunchArgument(
        'max_new_tokens',
        default_value='256',
        description='Maximum number of new tokens to generate'
    )
    
    vlm_server_node = Node(
        package='serenade_ros2',
        executable='vlm_server_node',
        name='vlm_server_node',
        output='screen',
        parameters=[
            {
                'image_topic': LaunchConfiguration('image_topic'),
                'model_name': LaunchConfiguration('model_name'),
                'max_new_tokens': LaunchConfiguration('max_new_tokens'),
            }
        ]
    )
    
    return LaunchDescription([
        image_topic_arg,
        model_name_arg,
        max_new_tokens_arg,
        vlm_server_node,
    ])
