from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
import os

def generate_launch_description():
    # Launch arguments
    model_name_arg = DeclareLaunchArgument(
        'model_name',
        default_value='Qwen/Qwen3-VL-4B-Instruct',
        description='VLM model name'
    )
    
    max_new_tokens_arg = DeclareLaunchArgument(
        'max_new_tokens',
        default_value='256',
        description='Maximum new tokens for VLM'
    )
    
    depth_model_arg = DeclareLaunchArgument(
        'depth_model',
        default_value='depth-anything/DA3-SMALL',
        description='Depth Anything model name'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Device for depth model (cuda or cpu)'
    )
    
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/image_raw',
        description='Input image topic'
    )
    
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/camera/camera_info',
        description='Camera info topic'
    )
    
    launch_rviz_arg = DeclareLaunchArgument(
        'launch_rviz',
        default_value='true',
        description='Whether to launch RViz2'
    )
    
    # Get package paths
    depth_pkg = FindPackageShare('depth_anything_3_ros2')
    yolo_pkg = FindPackageShare('yolo_world_ros2')
    serenade_pkg = FindPackageShare('serenade_ros2')
    
    # Depth Anything launch
    depth_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([depth_pkg, 'launch', 'depth_anything_3.launch.py'])
        ),
        launch_arguments={
            'image_topic': LaunchConfiguration('image_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'model_name': LaunchConfiguration('depth_model'),
            'device': LaunchConfiguration('device')
        }.items()
    )
    
    # YOLO World launch
    yolo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([yolo_pkg, 'launch', 'yolo_world_ros2_launch.py'])
        ),
        launch_arguments={
            'color_image': LaunchConfiguration('image_topic'),
            'depth_image': '/depth_anything_3/depth',
            'color_camerainfo': LaunchConfiguration('camera_info_topic')
        }.items()
    )
    
    # VLM Server launch
    vlm_server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([serenade_pkg, 'launch', 'vlm_server.launch.py'])
        ),
        launch_arguments={
            'image_topic': '/yolo_world/annotated_image',
            'model_name': LaunchConfiguration('model_name'),
            'max_new_tokens': LaunchConfiguration('max_new_tokens')
        }.items()
    )
    
    # Chatbot launch
    chatbot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([serenade_pkg, 'launch', 'chatbot.launch.py'])
        )
    )
    
    # Walker launch
    walker_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([serenade_pkg, 'launch', 'walker.launch.py'])
        )
    )
    
    # RViz2 - only launch if requested
    rviz_config_path = PathJoinSubstitution([
        depth_pkg, 'rviz', 'depth_view.rviz'
    ])
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        condition=IfCondition(LaunchConfiguration('launch_rviz')),
        output='screen'
    )
    
    return LaunchDescription([
        # Launch arguments
        model_name_arg,
        max_new_tokens_arg,
        depth_model_arg,
        device_arg,
        image_topic_arg,
        camera_info_topic_arg,
        launch_rviz_arg,
        
        # Launch components
        depth_launch,
        yolo_launch,
        vlm_server_launch,
        chatbot_launch,
        walker_launch,
        
        # RViz
        rviz_node
    ])