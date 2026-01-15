"""
Flask API module for HTTP endpoints.

This module provides the Flask application with endpoints for
image processing, depth estimation, and text generation.
"""

import datetime
from typing import Optional

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from ros_api import TrackingDataClient, TrackingResult

from ..config import config
from ..depth.comparison import plot_depth_comparison, save_da3_depth_with_ros_keypoints
from ..depth.generator import generate_depth_map
from ..depth.pointcloud import PointCloudValidator
from ..image.convert import cv2_to_pil
from ..models.model_loader import ModelManager
from ..text.generator import generate_text_stream


def create_flask_app(model_manager: Optional[ModelManager] = None) -> Flask:
    """
    Create and configure Flask application.

    Args:
        model_manager: Pre-initialized ModelManager instance. If None, creates new one.

    Returns:
        Configured Flask application instance
    """
    app = Flask(__name__)
    CORS(app)  # Enable CORS for cross-origin requests

    # Initialize model manager if not provided
    if model_manager is None:
        model_manager = ModelManager()
        model_manager.load_all_models()

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        return jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.datetime.now().isoformat(),
                "models": {
                    "vlm_loaded": model_manager.is_vlm_loaded(),
                    "da3_loaded": model_manager.is_da3_loaded(),
                    "sam3_loaded": model_manager.is_sam3_loaded(),
                },
            }
        )

    @app.route("/generate", methods=["POST"])
    def generate():
        """
        Main generation endpoint.

        Processes image from ROS and streams text response.
        """
        # 1. Parse request parameters
        data = request.json or {}
        text_query = data.get("text", "Describe this image")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # 2. Create new ROS client for this request
            print(f"[{timestamp}] 🔍 Creating new ROS client instance...")
            ros_client = TrackingDataClient(
                server_ip=config.ROS_SERVER_IP,
                port=config.ROS_SERVER_PORT,
                enable_log=config.ROS_CLIENT_ENABLE_LOG,
            )

            # 3. Get tracking data from ROS
            print(f"[{timestamp}] 🔍 Getting tracking data from ROS...")
            tracking_result: Optional[TrackingResult] = (
                ros_client.complete_tracking_pipeline()
            )

            if not tracking_result or not tracking_result.success:
                error_msg = "Failed to get tracking data from ROS"
                if tracking_result and not tracking_result.success:
                    error_msg = "ROS service returned failure status"
                raise Exception(error_msg)

            if tracking_result.current_image is None:
                raise Exception("No image in tracking result")

            # 4. Convert OpenCV image to PIL Image and save
            cv_image = tracking_result.current_image
            (_, image_save_path) = cv2_to_pil(cv_image)

            # 5. Stream text generation response
            return Response(
                generate_text_stream(
                    text_query, image_save_path, timestamp, model_manager
                ),
                mimetype="text/event-stream",
            )

        except Exception as e:
            error_msg = f"[{timestamp}] ❌ Error: {str(e)}"
            print(error_msg)
            return jsonify({"error": error_msg}), 500

    @app.route("/depth", methods=["POST"])
    def depth_only():
        """
        Depth-only endpoint for generating depth maps without text generation.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Create new ROS client
            print(f"[{timestamp}] 🔍 Creating new ROS client for depth generation...")
            ros_client = TrackingDataClient(
                server_ip=config.ROS_SERVER_IP,
                port=config.ROS_SERVER_PORT,
                enable_log=config.ROS_CLIENT_ENABLE_LOG,
            )

            # Get tracking data from ROS
            print(f"[{timestamp}] 🔍 Getting tracking data from ROS...")
            tracking_result: Optional[TrackingResult] = (
                ros_client.complete_tracking_pipeline()
            )

            if not tracking_result or not tracking_result.success:
                error_msg = "Failed to get tracking data from ROS"
                if tracking_result and not tracking_result.success:
                    error_msg = "ROS service returned failure status"
                raise Exception(error_msg)

            if tracking_result.current_image is None:
                raise Exception("No image in tracking result")

            # Convert OpenCV image to PIL Image and save
            cv_image = tracking_result.current_image
            (_, image_save_path) = cv2_to_pil(cv_image)

            # Get image dimensions
            image_shape = (int(cv_image.shape[0]), int(cv_image.shape[1]))  # (h, w)

            # Generate depth map
            print(f"[{timestamp}] 📊 Generating DA3 depth map...")
            da3_depth_map = generate_depth_map(
                image_save_path, image_shape, model_manager
            )

            # Create visualizations
            camera_point_cloud = tracking_result.tracked_points_camera
            world_point_cloud = tracking_result.tracked_points_world

            if camera_point_cloud is not None and camera_point_cloud.size > 0:
                plot_depth_comparison(
                    camera_point_cloud, da3_depth_map, timestamp, image_shape
                )
                save_da3_depth_with_ros_keypoints(
                    da3_depth_map, camera_point_cloud, timestamp, image_shape
                )

            # Validate point cloud using PointCloudValidator
            if (
                camera_point_cloud is not None
                and camera_point_cloud.size > 0
                and world_point_cloud is not None
                and world_point_cloud.size > 0
                and tracking_result.camera_pose is not None
            ):
                validator = PointCloudValidator(tracking_result.camera_pose)

                error_metrics, point_errors = validator.validate_point_cloud(
                    camera_point_cloud, world_point_cloud
                )

                print(f"[{timestamp}] 📊 Point cloud validation results:")
                print(f"  • Point count: {error_metrics['point_count']}")
                print(f"  • Mean error: {error_metrics['mean_error']:.6f}")
                print(f"  • Max error: {error_metrics['max_error']:.6f}")
                print(f"  • Min error: {error_metrics['min_error']:.6f}")
                print(f"  • RMSE: {error_metrics['rmse']:.6f}")

            # Return success response with file paths
            response_data = {
                "timestamp": timestamp,
                "image_path": image_save_path,
                "depth_map_shape": da3_depth_map.shape,
                "depth_plot_path": config.get_depth_plot_path(timestamp),
                "da3_depth_path": config.get_da3_depth_path(timestamp),
                "da3_depth_keypoints_path": config.get_da3_depth_with_keypoints_path(
                    timestamp
                ),
            }

            return jsonify(response_data)

        except Exception as e:
            error_msg = f"[{timestamp}] ❌ Depth generation error: {str(e)}"
            print(error_msg)
            return jsonify({"error": error_msg}), 500

    @app.route("/models", methods=["GET"])
    def list_models():
        """List available models and their status."""
        return jsonify(
            {
                "vlm": {
                    "loaded": model_manager.is_vlm_loaded(),
                    "default": config.MODEL_VLM_DEFAULT,
                    "available": [
                        config.MODEL_QWEN_2B,
                        config.MODEL_QWEN_4B,
                        config.MODEL_QWEN_8B,
                        config.MODEL_SMOLVLM,
                    ],
                },
                "da3": {
                    "loaded": model_manager.is_da3_loaded(),
                    "default": config.MODEL_DA3_DEFAULT,
                    "available": [
                        config.MODEL_DA3_LARGE,
                        config.MODEL_DA3_NESTED,
                    ],
                },
                "sam3": {
                    "loaded": model_manager.is_sam3_loaded(),
                    "default": config.MODEL_SAM3_PATH,
                },
            }
        )

    return app


def run_server(
    host: str = config.FLASK_HOST,
    port: int = config.FLASK_PORT,
    debug: bool = False,
    model_manager: Optional[ModelManager] = None,
) -> None:
    """
    Run the Flask server.

    Args:
        host: Server host address
        port: Server port
        debug: Enable debug mode
        model_manager: Pre-initialized ModelManager instance
    """
    app = create_flask_app(model_manager)

    print(
        f"\n[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] 🚀 "
        f"Flask server starting at: http://{host}:{port}"
    )
    print(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] 📍 Endpoints:")
    print("  • POST /generate    - Text-generation pipeline")
    print("  • POST /depth       - Depth-only pipeline")
    print("  • GET  /health      - Health check")
    print("  • GET  /models      - List available models")

    app.run(host=host, port=port, threaded=True, debug=debug)
