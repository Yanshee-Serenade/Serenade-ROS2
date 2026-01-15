"""
Run all model clients for testing.

Tests 5 clients:
1. VLM Client - Vision Language Model inference
2. DA3 Client - Depth Anything 3 depth estimation
3. SAM3 Client - SAM3 segmentation
4. Comparison Client - Depth comparison between DA3 and ORB-SLAM3
5. Pointcloud Client - Point cloud validation
"""

import argparse
import datetime
import sys

import cv2

from client.comparison_client import run_comparison_client
from client.config import client_config
from client.da3_client import DA3Client
from client.pointcloud_client import run_pointcloud_client
from client.sam3_client import SAM3Client
from client.vlm_client import VLMClient
from ros_api import TrackingDataClient


def get_image_from_ros() -> tuple[str, tuple[int, int]]:
    """Get image from ROS server and save it."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n[{timestamp}] 📡 Getting image from ROS server...")

    ros_client = TrackingDataClient(
        server_ip=client_config.ROS_HOST,
        port=client_config.ROS_PORT,
        enable_log=False,
    )

    tracking_result = ros_client.complete_tracking_pipeline()
    if not tracking_result or not tracking_result.success:
        raise Exception("Failed to get tracking data from ROS")

    if tracking_result.current_image is None:
        raise Exception("No image in tracking result")

    # Save image
    image_path = client_config.get_image_path(timestamp)
    cv2.imwrite(image_path, tracking_result.current_image)
    image_shape = (
        tracking_result.current_image.shape[0],
        tracking_result.current_image.shape[1],
    )

    print(f"[{timestamp}] ✅ Image saved: {image_path}")
    return image_path, image_shape


def test_vlm_client(image_path: str) -> bool:
    """Test VLM client."""
    print("\n" + "=" * 60)
    print("1. Testing VLM Client (Vision Language Model)")
    print("=" * 60)

    try:
        client = VLMClient()
        text_query = "请简要描述图片内容"
        print(f"🤖 Query: {text_query}")
        print(f"📷 Image: {image_path}")
        print("📝 Response: ", end="", flush=True)

        full_response = ""
        for chunk in client.generate(image_path, text_query):
            print(chunk, end="", flush=True)
            full_response += chunk

        print("\n✅ VLM client test PASSED")
        return True
    except Exception as e:
        print(f"\n❌ VLM client test FAILED: {e}")
        return False


def test_da3_client(image_path: str) -> bool:
    """Test DA3 client."""
    print("\n" + "=" * 60)
    print("2. Testing DA3 Client (Depth Anything 3)")
    print("=" * 60)

    try:
        client = DA3Client()
        print(f"📊 Requesting depth inference for: {image_path}")

        prediction = client.inference(image_path)

        if prediction:
            print("✅ DA3 inference completed")
            print(f"  • Depth shape: {prediction.depth.shape}")
            print(f"  • Depth dtype: {prediction.depth.dtype}")
            print(
                f"  • Depth range: [{prediction.depth.min():.3f}, {prediction.depth.max():.3f}]"
            )
            print(f"  • Conf shape: {prediction.conf.shape}")
            print(f"  • Extrinsics shape: {prediction.extrinsics.shape}")
            print(f"  • Intrinsics shape: {prediction.intrinsics.shape}")
            print(f"  • Processed images shape: {prediction.processed_images.shape}")
            print("✅ DA3 client test PASSED")
            return True
        else:
            print("❌ DA3 client test FAILED: No prediction returned")
            return False
    except Exception as e:
        print(f"❌ DA3 client test FAILED: {e}")
        return False


def test_sam3_client(image_path: str) -> bool:
    """Test SAM3 client."""
    print("\n" + "=" * 60)
    print("3. Testing SAM3 Client (Segmentation)")
    print("=" * 60)

    try:
        client = SAM3Client()
        prompt = "wheel"
        print(f"🎯 Requesting segmentation for: {image_path}")
        print(f"💬 Prompt: {prompt}")

        state = client.inference(image_path, prompt)

        if state:
            print("✅ SAM3 inference completed")
            print(f"  • Original size: {state.original_height}x{state.original_width}")
            print(f"  • Masks logits shape: {state.masks_logits.shape}")
            print(f"  • Masks logits dtype: {state.masks_logits.dtype}")
            print(f"  • Masks shape: {state.masks.shape}")
            print(f"  • Masks dtype: {state.masks.dtype}")
            print(f"  • Boxes shape: {state.boxes.shape}")
            print(f"  • Scores shape: {state.scores.shape}")
            print(f"  • Number of segments: {len(state.scores)}")
            print("✅ SAM3 client test PASSED")
            return True
        else:
            print("❌ SAM3 client test FAILED: No state returned")
            return False
    except Exception as e:
        print(f"❌ SAM3 client test FAILED: {e}")
        return False


def test_comparison_client() -> bool:
    """Test comparison client."""
    print("\n" + "=" * 60)
    print("4. Testing Comparison Client (DA3 + ROS)")
    print("=" * 60)

    try:
        success = run_comparison_client()
        if success:
            print("✅ Comparison client test PASSED")
        else:
            print("❌ Comparison client test FAILED")
        return success
    except Exception as e:
        print(f"❌ Comparison client test FAILED: {e}")
        return False


def test_pointcloud_client() -> bool:
    """Test pointcloud validation client."""
    print("\n" + "=" * 60)
    print("5. Testing Pointcloud Validation Client")
    print("=" * 60)

    try:
        success = run_pointcloud_client()
        if success:
            print("✅ Pointcloud client test PASSED")
        else:
            print("❌ Pointcloud client test FAILED")
        return success
    except Exception as e:
        print(f"❌ Pointcloud client test FAILED: {e}")
        return False


def main():
    """Main entry point for client testing."""
    parser = argparse.ArgumentParser(description="Test all model clients")
    parser.add_argument(
        "--test",
        choices=["all", "vlm", "da3", "sam3", "comparison", "pointcloud"],
        default="all",
        help="Which client(s) to test (default: all)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Use specific image path instead of getting from ROS",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🧪 Model Client Testing Suite")
    print("=" * 60)

    results = {}

    try:
        # Get image
        if args.image:
            image_path = args.image
            print(f"\n📷 Using provided image: {image_path}")
            image_shape = None
        else:
            image_path, image_shape = get_image_from_ros()

        # Run tests based on selection
        if args.test in ["all", "vlm"]:
            results["vlm"] = test_vlm_client(image_path)

        if args.test in ["all", "da3"]:
            results["da3"] = test_da3_client(image_path)

        if args.test in ["all", "sam3"]:
            results["sam3"] = test_sam3_client(image_path)

        if args.test in ["all", "comparison"]:
            results["comparison"] = test_comparison_client()

        if args.test in ["all", "pointcloud"]:
            results["pointcloud"] = test_pointcloud_client()

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    print(f"\nTotal tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")

    print("\nDetailed results:")
    test_names = {
        "vlm": "VLM Client",
        "da3": "DA3 Client",
        "sam3": "SAM3 Client",
        "comparison": "Comparison Client",
        "pointcloud": "Pointcloud Client",
    }

    for test_key, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_names[test_key]}")

    print("\n" + "=" * 60)
    if passed == total:
        print("✅ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
