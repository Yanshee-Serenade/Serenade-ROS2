#!/usr/bin/env python3
"""
Minimal LLM Client Test Script

Tests only the 4 async APIs from the Flask server:
1. GET /health - Health check
2. GET /models - List models
3. POST /generate - Text generation (streaming)
4. POST /depth - Depth map generation
"""

import asyncio
import sys
from typing import Dict

import aiohttp


async def test_health(session: aiohttp.ClientSession, base_url: str) -> bool:
    """Test health check endpoint."""
    try:
        async with session.get(f"{base_url}/health") as response:
            if response.status == 200:
                data = await response.json()
                print(f"✓ Health check: {data['status']}")
                print(
                    f"  Models: VLM={data['models']['vlm_loaded']}, "
                    f"DA3={data['models']['da3_loaded']}, "
                    f"SAM3={data['models']['sam3_loaded']}"
                )
                return True
            else:
                print(f"✗ Health check failed: HTTP {response.status}")
                return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False


async def test_models(session: aiohttp.ClientSession, base_url: str) -> bool:
    """Test models listing endpoint."""
    try:
        async with session.get(f"{base_url}/models") as response:
            if response.status == 200:
                data = await response.json()
                print("✓ Models listing:")
                for model_type, info in data.items():
                    print(
                        f"  {model_type}: loaded={info['loaded']}, "
                        f"default={info['default']}"
                    )
                return True
            else:
                print(f"✗ Models listing failed: HTTP {response.status}")
                return False
    except Exception as e:
        print(f"✗ Models listing error: {e}")
        return False


async def test_generate(session: aiohttp.ClientSession, base_url: str) -> bool:
    """Test text generation endpoint."""
    try:
        payload = {"text": "你看到了什么？请简短回答"}
        async with session.post(f"{base_url}/generate", json=payload) as response:
            if response.status == 200:
                print("✓ Text generation (streaming response)")
                # Read Server-Sent Events (SSE) streaming response
                buffer = ""
                async for line in response.content:
                    if line:
                        line_str = line.decode("utf-8")
                        buffer += line_str
                        # Look for SSE data lines
                        if line_str.startswith("data: "):
                            data = line_str[6:].strip()
                            if data:
                                print(f"  Data: {data}")
                return True
            else:
                print(f"✗ Text generation failed: HTTP {response.status}")
                return False
    except Exception as e:
        print(f"✗ Text generation error: {e}")
        return False


async def test_depth(session: aiohttp.ClientSession, base_url: str) -> bool:
    """Test depth map generation endpoint."""
    try:
        async with session.post(f"{base_url}/depth") as response:
            if response.status == 200:
                data = await response.json()
                print(f"✓ Depth generation: {data['timestamp']}")
                print(f"  Image: {data['image_path']}")
                print(f"  Depth shape: {data['depth_map_shape']}")
                return True
            else:
                print(f"✗ Depth generation failed: HTTP {response.status}")
                return False
    except Exception as e:
        print(f"✗ Depth generation error: {e}")
        return False


async def run_all_tests(base_url: str = "http://localhost:21122") -> Dict[str, bool]:
    """Run all 4 API tests."""
    print(f"Testing Flask APIs at: {base_url}")
    print("=" * 50)

    timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
    results = {}

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Test 1: Health check
        print("\n1. Testing /health endpoint...")
        results["health"] = await test_health(session, base_url)

        # Test 2: Models listing
        print("\n2. Testing /models endpoint...")
        results["models"] = await test_models(session, base_url)

        # Test 3: Text generation
        print("\n3. Testing /generate endpoint...")
        results["generate"] = await test_generate(session, base_url)
        # results["generate"] = True

        # Test 4: Depth generation
        print("\n4. Testing /depth endpoint...")
        results["depth"] = await test_depth(session, base_url)

    return results


def print_summary(results: Dict[str, bool]):
    """Print test summary."""
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    print(f"\nTotal tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")

    print("\nDetailed results:")
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")

    print("\n" + "=" * 50)
    if passed == total:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Minimal Flask API Test")
    parser.add_argument(
        "--base-url",
        default="http://localhost:21122",
        help="Base URL of the Flask server (default: http://localhost:21122)",
    )

    args = parser.parse_args()

    try:
        results = await run_all_tests(args.base_url)
        print_summary(results)

        # Exit with appropriate code
        if all(results.values()):
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
