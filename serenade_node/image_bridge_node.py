#!/usr/bin/env python3
# ros2_tcp_image_client.py
"""
ROS2 TCP Image Client
Receives images over TCP and publishes to ROS2.
Expects: 53-byte header + image data
"""
import rclpy
from rclpy.node import Node
import socket
import struct
import threading
import time
from sensor_msgs.msg import Image
from std_msgs.msg import Header

class ImageBridgeNode(Node):
    def __init__(self, host='127.0.0.1', port=21121):
        super().__init__('image_bridge_node')
        
        # ROS2 publisher
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        self.publisher = self.create_publisher(
            Image,
            '/camera/image_slow',
            qos
        )
        
        # Connection
        self.host = host
        self.port = port
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        self.last_log_time = time.time()
        
        # Start receiver thread
        self.receiver_thread = threading.Thread(target=self.receive_loop)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()
        
        self.get_logger().info(f"TCP Image Client starting, connecting to {host}:{port}")
    
    def receive_loop(self):
        """Main receive loop"""
        while rclpy.ok():
            sock = None
            try:
                # Connect to server
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(5.0)
                sock.connect((self.host, self.port))
                sock.settimeout(None)  # Blocking
                
                self.get_logger().info("Connected to server")
                
                # Main receive loop
                while rclpy.ok():
                    # 1. Receive header (53 bytes)
                    header = self.recv_exact(sock, 53)
                    if not header:
                        break
                    
                    # 2. Parse header
                    (sec, nsec, height, width, 
                     encoding_bytes, step, is_bigendian, data_size) = struct.unpack(
                        '!QQII16sIBQ', header
                    )
                    
                    # 3. Get encoding string (strip null bytes)
                    encoding = encoding_bytes.split(b'\0')[0].decode('ascii', errors='ignore')
                    
                    # 4. Receive image data
                    image_data = self.recv_exact(sock, data_size)
                    if not image_data:
                        break
                    
                    # 5. Publish to ROS2
                    self.publish_image(
                        sec, nsec, height, width, encoding,
                        step, bool(is_bigendian), image_data
                    )
                    
            except socket.timeout:
                self.get_logger().warn("Connection timeout")
            except ConnectionRefusedError:
                self.get_logger().warn(f"Connection refused to {self.host}:{self.port}")
                time.sleep(1)
            except Exception as e:
                self.get_logger().error(f"Receive error: {str(e)}")
                time.sleep(1)
            finally:
                if sock:
                    try:
                        sock.close()
                    except:
                        pass
    
    def recv_exact(self, sock, n):
        """Receive exactly n bytes"""
        data = bytearray()
        while len(data) < n:
            try:
                chunk = sock.recv(n - len(data))
                if not chunk:
                    return None
                data.extend(chunk)
            except socket.timeout:
                return None
            except Exception as e:
                self.get_logger().warn(f"Recv error: {e}")
                return None
        return bytes(data)
    
    def publish_image(self, sec, nsec, height, width, encoding, step, is_bigendian, data):
        """Create and publish ROS2 image message"""
        # Create message
        msg = Image()
        
        # Set header
        msg.header = Header()
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nsec
        msg.header.frame_id = f"tcp_frame_{self.frame_count}"
        
        # Set image properties
        msg.height = height
        msg.width = width
        msg.encoding = encoding
        msg.step = step
        msg.is_bigendian = is_bigendian
        msg.data = data
        
        # Publish
        self.publisher.publish(msg)
        
        # Update stats
        self.frame_count += 1
        
        # Log every 5 seconds
        now = time.time()
        if now - self.last_log_time > 5.0:
            fps = self.frame_count / (now - self.start_time)
            self.get_logger().info(
                f"FPS: {fps:.1f} | "
                f"Frames: {self.frame_count} | "
                f"Size: {len(data)/1024:.1f}KB | "
                f"Res: {width}x{height}"
            )
            self.last_log_time = now
    
    def destroy_node(self):
        """Clean shutdown"""
        self.get_logger().info("TCP Image Client shutting down")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    node = ImageBridgeNode(host='127.0.0.1', port=21121)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()