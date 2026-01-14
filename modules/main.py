"""
Main module as the entry point for the Yanshee server.
Coordinates initialization and starts the Flask server.
"""

import datetime
import os
import sys
from typing import Optional

# Add modules directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules
from modules import app, config, models


def initialize_models() -> bool:
    """
    Initialize all AI models.

    Returns:
        True if all required models initialized successfully, False otherwise
    """
    print("🔍 Initializing AI models...")

    # Initialize VLM model
    vlm_model, vlm_processor = models.load_model_vlm(config.MODEL_VLM_DEFAULT)
    if vlm_model is None or vlm_processor is None:
        print("❌ Failed to initialize VLM model")
        return False

    # Initialize DA3 model
    da3_model = models.load_model_da3(config.MODEL_DA3_DEFAULT)
    if da3_model is None:
        print("❌ Failed to initialize DA3 model")
        return False

    # Optional: Initialize SAM3 model
    # sam3_model, sam3_processor = models.load_model_sam3(config.MODEL_SAM3_PATH)
    # if sam3_model is None or sam3_processor is None:
    #     print("⚠️ Failed to initialize SAM3 model (optional)")

    print("✅ All required models initialized successfully")
    return True


def cleanup() -> None:
    """
    Clean up resources before exit.
    """
    print("\n🧹 Cleaning up resources...")

    # Clean up models
    models.cleanup_models()

    print("✅ Resources cleaned up")


def main() -> None:
    """
    Main entry point for the Yanshee server.
    """
    try:
        # Print startup banner
        print("=" * 60)
        print("🚀 Yanshee Server - Starting Up")
        print("=" * 60)
        print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python version: {sys.version}")
        print(f"Working directory: {os.getcwd()}")
        print("-" * 60)

        # Step 1: Initialize AI models
        if not initialize_models():
            print("❌ Model initialization failed. Exiting...")
            sys.exit(1)

        # Step 2: Initialize Flask application
        print("\n🔍 Initializing Flask application...")
        flask_app = app.init_flask_app()
        if flask_app is None:
            print("❌ Flask application initialization failed. Exiting...")
            sys.exit(1)

        # Step 3: Start Flask server
        print("\n" + "=" * 60)
        print("✅ Initialization complete")
        print("=" * 60)

        # Register cleanup handler
        import atexit

        atexit.register(cleanup)

        # Start the server
        app.run_app(
            host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            debug=False,  # Production mode
            threaded=True,
        )

    except KeyboardInterrupt:
        print("\n\n⚠️ Received keyboard interrupt. Shutting down...")
        cleanup()
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback

        traceback.print_exc()
        cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
