"""
Launch file for the VLM agent node
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for VLM agent node"""
    
    # Declare launch arguments
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/image_raw',
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
    
    use_history_arg = DeclareLaunchArgument(
        'use_history',
        default_value='false',
        description='Whether to load and use conversation history'
    )
    
    agent_node = Node(
        package='serenade_ros2',
        executable='agent_node',
        name='agent_node',
        output='screen',
        parameters=[
            {
                'image_topic': LaunchConfiguration('image_topic'),
                'model_name': LaunchConfiguration('model_name'),
                'max_new_tokens': LaunchConfiguration('max_new_tokens'),
                'use_history': LaunchConfiguration('use_history'),
            }
        ]
    )
    
    return LaunchDescription([
        image_topic_arg,
        model_name_arg,
        max_new_tokens_arg,
        use_history_arg,
        agent_node,
    ])
