"""
Modules package for Yanshee Model Server.

This package contains modularized components for the Yanshee server,
including configuration, model management, ROS client, depth processing,
text generation, Flask application, and main entry point.
"""

from .app import (
    create_error_response,
    get_app,
    init_flask_app,
    run_app,
)
from .config import (
    DA3_DEPTH_SAVE_PREFIX,
    DA3_DEPTH_WITH_KEYPOINTS_SAVE_PREFIX,
    DEPTH_EXPORT_FORMAT,
    DEPTH_PLOT_SAVE_PREFIX,
    DEPTH_PROCESS_METHOD,
    DEPTH_PROCESS_RES,
    FLASK_HOST,
    FLASK_PORT,
    IMAGE_SAVE_PREFIX,
    MAX_NEW_TOKENS,
    MODEL_DA3_DEFAULT,
    MODEL_DA3_LARGE,
    MODEL_DA3_NESTED,
    MODEL_QWEN_2B,
    MODEL_QWEN_4B,
    MODEL_QWEN_8B,
    MODEL_SAM3_PATH,
    MODEL_SMOLVLM,
    MODEL_VLM_DEFAULT,
    ROS_SERVER_IP,
    ROS_SERVER_PORT,
    ImageShape,
    PointCloud,
)
from .depth_processing import (
    calculate_depth_statistics,
    generate_depth_map,
    plot_depth_comparison,
    save_da3_depth_with_keypoints,
    save_depth_visualization,
    visualize_depth_map,
)
from .main import main
from .models import (
    cleanup_models,
    get_da3_model,
    get_sam3_model,
    get_sam3_processor,
    get_vlm_model,
    get_vlm_processor,
    is_da3_loaded,
    is_sam3_loaded,
    is_vlm_loaded,
    load_model_da3,
    load_model_sam3,
    load_model_vlm,
)
from .ros_client import (
    extract_depth_at_pixels,
    get_current_timestamp,
    get_image_from_ros,
    init_tracking_client,
    validate_point_cloud,
)
from .text_generation import (
    create_streamer,
    format_error_response,
    format_success_response,
    generate_text_batch,
    generate_text_stream,
    prepare_vlm_input,
    validate_text_query,
)

__all__ = [
    # Configuration
    'MODEL_QWEN_8B',
    'MODEL_QWEN_4B',
    'MODEL_QWEN_2B',
    'MODEL_SMOLVLM',
    'MODEL_DA3_LARGE',
    'MODEL_DA3_NESTED',
    "MODEL_QWEN_8B",
    "MODEL_QWEN_4B",
    "MODEL_QWEN_2B",
    "MODEL_SMOLVLM",
    "MODEL_DA3_LARGE",
    "MODEL_DA3_NESTED",
    "MODEL_VLM_DEFAULT",
    "MODEL_DA3_DEFAULT",
    "MODEL_SAM3_PATH",
    "ROS_SERVER_IP",
    "ROS_SERVER_PORT",
    "FLASK_HOST",
    "FLASK_PORT",
    "MAX_NEW_TOKENS",
    "IMAGE_SAVE_PREFIX",
    "DEPTH_PLOT_SAVE_PREFIX",
    "DA3_DEPTH_SAVE_PREFIX",
    "DA3_DEPTH_WITH_KEYPOINTS_SAVE_PREFIX",
    "DEPTH_PROCESS_RES",
    "DEPTH_PROCESS_METHOD",
    "DEPTH_EXPORT_FORMAT",
    "ImageShape",
    "PointCloud",
    # Models
    "load_model_vlm",
    "load_model_da3",
    "load_model_sam3",
    "get_vlm_model",
    "get_vlm_processor",
    "get_da3_model",
    "get_sam3_model",
    "get_sam3_processor",
    "is_vlm_loaded",
    "is_da3_loaded",
    "is_sam3_loaded",
    "cleanup_models",
    # ROS Client
    "init_tracking_client",
    "get_image_from_ros",
    "validate_point_cloud",
    "extract_depth_at_pixels",
    "get_current_timestamp",
    # Depth Processing
    "generate_depth_map",
    "visualize_depth_map",
    "save_depth_visualization",
    "plot_depth_comparison",
    "save_da3_depth_with_keypoints",
    "calculate_depth_statistics",
    # Text Generation
    "prepare_vlm_input",
    "create_streamer",
    "generate_text_stream",
    "generate_text_batch",
    "validate_text_query",
    "format_error_response",
    "format_success_response",
    # Flask Application
    "init_flask_app",
    "get_app",
    "run_app",
    "create_error_response",
    # Main
    "main",
]

__version__ = "1.0.0"
__author__ = "Yanshee Model Server Team"
__description__ = "Modularized server for Yanshee robot with AI model integration"
