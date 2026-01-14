"""
Flask application module for API endpoints.
Handles Flask app initialization, routing, and request processing.
"""

import datetime
import json
from typing import Dict, Optional, Tuple

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

# Import modules
from . import config, depth_processing, models, ros_client, text_generation

# Global Flask app instance
app: Optional[Flask] = None


def init_flask_app() -> Flask:
    """
    Initialize Flask application with CORS and routes.

    Returns:
        Initialized Flask application instance
    """
    global app

    # Create Flask app
    app = Flask(__name__)
    CORS(app)  # Enable CORS for cross-origin requests

    # Register routes
    register_routes(app)

    print("✅ Flask application initialized")
    print(f"   Host: {config.FLASK_HOST}")
    print(f"   Port: {config.FLASK_PORT}")
    print("   CORS: Enabled")

    return app


def register_routes(flask_app: Flask) -> None:
    """
    Register all API routes.

    Args:
        flask_app: Flask application instance
    """

    # Health check endpoint
    @flask_app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        return jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.datetime.now().isoformat(),
                "models": {
                    "vlm": models.is_vlm_loaded(),
                    "da3": models.is_da3_loaded(),
                    "sam3": models.is_sam3_loaded(),
                },
            }
        )

    # Main generation endpoint
    @flask_app.route("/generate", methods=["POST"])
    def generate():
        """Generate text from image with depth processing."""
        return handle_generate_request()

    # Test endpoint for debugging
    @flask_app.route("/test", methods=["GET"])
    def test_endpoint():
        """Test endpoint for debugging."""
        return jsonify(
            {
                "message": "Server is running",
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )


def handle_generate_request() -> Response:
    """
    Handle /generate endpoint request.

    Returns:
        Flask Response with streaming text or error
    """
    # Parse request parameters
    data = request.json or {}
    text_query = data.get("text", "Describe this image")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Validate text query
        is_valid, error_msg = text_generation.validate_text_query(text_query)
        if not is_valid:
            return create_error_response(error_msg, 400)

        # Step 1: Initialize ROS client for this request
        print(f"[{timestamp}] 🔍 Creating new ROS client instance...")
        ros_client_instance = ros_client.init_tracking_client(enable_log=False)
        if not ros_client_instance:
            raise Exception("Failed to create ROS client instance")

        # Step 2: Get image and point cloud data from ROS
        print(f"[{timestamp}] 🔍 Getting image and point cloud data from ROS...")
        pil_image, image_path, camera_point_cloud, world_point_cloud, cv_image = (
            ros_client.get_image_from_ros(ros_client_instance, timestamp)
        )

        if not pil_image:
            raise Exception(image_path)  # image_path contains error message

        # Step 3: Get original image dimensions
        if cv_image is None:
            raise Exception("OpenCV image is None")
        image_shape = cv_image.shape[:2]  # (height, width)
        height, width = image_shape
        image_shape_tuple = (int(height), int(width))
        print(f"[{timestamp}] 📏 Original image dimensions: {image_shape_tuple}")

        # Step 4: Generate DA3 depth map
        print(f"[{timestamp}] 📊 Generating DA3 depth map...")
        da3_depth_map = depth_processing.generate_depth_map(
            image_path, image_shape_tuple
        )

        if da3_depth_map is None:
            raise Exception("Failed to generate DA3 depth map")

        # Step 5: Process depth data
        print(f"[{timestamp}] 🔬 Processing depth data...")

        # Plot depth comparison
        if camera_point_cloud is not None:
            depth_processing.plot_depth_comparison(
                camera_point_cloud, da3_depth_map, timestamp, image_shape_tuple
            )

        # Save depth map with keypoints
        depth_processing.save_da3_depth_with_keypoints(
            da3_depth_map, camera_point_cloud, timestamp, image_shape_tuple
        )

        # Step 6: Stream text generation results
        print(f"[{timestamp}] 💬 Starting text generation...")
        return Response(
            text_generation.generate_text_stream(text_query, image_path, timestamp),
            mimetype="text/event-stream",
        )

    except Exception as e:
        error_msg = f"[{timestamp}] ❌ Error: {str(e)}"
        print(error_msg)
        return create_error_response(error_msg, 500)


def get_app() -> Optional[Flask]:
    """
    Get the Flask application instance.

    Returns:
        Flask app instance or None if not initialized
    """
    return app


def run_app(
    host: Optional[str] = None,
    port: Optional[int] = None,
    debug: bool = False,
    threaded: bool = True,
) -> None:
    """
    Run the Flask application.

    Args:
        host: Host to bind to (defaults to config.FLASK_HOST)
        port: Port to bind to (defaults to config.FLASK_PORT)
        debug: Enable debug mode
        threaded: Enable threading
    """
    if app is None:
        raise RuntimeError(
            "Flask application not initialized. Call init_flask_app() first."
        )

    run_host = host or config.FLASK_HOST
    run_port = port or config.FLASK_PORT

    print(
        f"\n[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] 🚀 Starting Flask server..."
    )
    print(f"   Address: http://{run_host}:{run_port}")
    print(f"   Debug: {debug}")
    print(f"   Threaded: {threaded}")

    app.run(
        host=run_host,
        port=run_port,
        debug=debug,
        threaded=threaded,
    )


def create_error_response(
    message: str,
    status_code: int = 500,
    details: Optional[Dict] = None,
) -> Tuple[Response, int]:
    """
    Create standardized error response.

    Args:
        message: Error message
        status_code: HTTP status code
        details: Additional error details

    Returns:
        Tuple of (Response, status_code)
    """
    response_data = {
        "error": message,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    if details:
        if details:
            response_data["details"] = details

    return jsonify(response_data), status_code
