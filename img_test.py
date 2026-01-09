import requests
import time
import sys

def test_connectivity():
    docker_ip = "localhost"  # 或容器实际IP
    port = 51121
    url = f"http://{docker_ip}:{port}/image"
    
    print(f"Testing connection to {url}")
    
    # 测试1: 简单连接
    print("1. Testing basic connection...")
    try:
        response = requests.get(url, timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Content-Type: {response.headers.get('Content-Type')}")
        print(f"   Content-Length: {response.headers.get('Content-Length')}")
        
        if response.status_code == 200:
            print(f"   ✓ Success! Received {len(response.content)} bytes")
            return True
        else:
            print(f"   ✗ Failed with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("   ✗ Connection refused")
    except requests.exceptions.Timeout:
        print("   ✗ Connection timeout")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    return False

def test_streaming():
    docker_ip = "localhost"
    port = 51121
    url = f"http://{docker_ip}:{port}/image"
    
    print("\n2. Testing image streaming...")
    
    try:
        response = requests.get(url, stream=True, timeout=5)
        
        if response.status_code == 200:
            print(f"   Starting download...")
            
            start_time = time.time()
            total_bytes = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    total_bytes += len(chunk)
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        speed = total_bytes / elapsed / 1024
                        sys.stdout.write(f"\r   Downloaded: {total_bytes:,} bytes ({speed:.1f} KB/s)")
                        sys.stdout.flush()
            
            print(f"\n   ✓ Download complete: {total_bytes:,} bytes")
            return True
        else:
            print(f"   ✗ Failed with status {response.status_code}")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    return False

def main():
    print("=" * 50)
    print("Docker Image Server Connectivity Test")
    print("=" * 50)
    
    # 先测试基本连接
    if not test_connectivity():
        print("\n[ERROR] Basic connection failed. Check:")
        print("  1. Docker container is running")
        print("  2. Port 51121 is exposed (docker run -p 51121:51121)")
        print("  3. Server is running inside container")
        print("  4. Network connectivity")
        return
    
    # 再测试流式传输
    test_streaming()
    
    print("\n" + "=" * 50)
    print("Test completed")
    print("=" * 50)

if __name__ == "__main__":
    main()