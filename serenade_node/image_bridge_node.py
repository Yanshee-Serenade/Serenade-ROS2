#!/usr/bin/env python3
# ros2_image_client.py
import rclpy
from rclpy.node import Node
import socket
import struct
import threading
import time
from sensor_msgs.msg import Image
import numpy as np

# ros1_bridge 无敌了，竟然让我手写这种东西
# 转发一个 5 Hz 的图像都能给我干到 4 Hz
# 真无敌了，改 QoS 也没用，static bridge 是假的
# https://answers.ros.org/question/380088/ 无人在意
class ImageBridgeNode(Node):
    def __init__(self, host='127.0.0.1', port=21121):
        super().__init__('image_bridge_node')
        
        # ROS2 publisher
        qos_policy = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE
        )
        self.publisher = self.create_publisher(
            Image, 
            '/camera/image_slow', 
            qos_policy
        )
        
        # Connection parameters
        self.host = host
        self.port = port
        self.reconnect_delay = 1.0  # seconds
        
        # Statistics
        self.frame_count = 0
        self.last_log_time = time.time()
        self.last_frame_time = None
        self.avg_latency = 0.0
        
        # Start connection thread
        self.connection_thread = threading.Thread(target=self.connection_loop)
        self.connection_thread.daemon = True
        self.connection_thread.start()
        
        self.get_logger().info(f"ROS2 Image Client connecting to {host}:{port}")
    
    def connection_loop(self):
        """Main connection and data reception loop"""
        sock = None
        
        while rclpy.ok():
            try:
                # Connect to server
                if sock is None:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle
                    sock.settimeout(5.0)
                    sock.connect((self.host, self.port))
                    self.get_logger().info(f"Connected to {self.host}:{self.port}")
                    sock.settimeout(None)  # Blocking receive
                
                # Receive data
                while rclpy.ok():
                    # First receive the fixed-size header (without variable encoding string)
                    # We'll get the encoding length first, then read the rest
                    header_prefix = self.recv_exact(sock, 40)  # First 40 bytes are fixed
                    if not header_prefix:
                        break
                    
                    # Unpack fixed part
                    (frame_id, ts_sec, ts_nsec, height, width, encoding_len) = \
                        struct.unpack('!Q Q Q I I I', header_prefix)
                    
                    # Now receive the variable part
                    variable_part = self.recv_exact(sock, encoding_len + 13)  # encoding + step(4) + is_bigendian(1) + data_len(8)
                    if not variable_part:
                        break
                    
                    # Unpack variable part
                    encoding = variable_part[:encoding_len].decode('utf-8')
                    offset = encoding_len
                    step = struct.unpack_from('!I', variable_part, offset)[0]; offset += 4
                    is_bigendian = struct.unpack_from('!B', variable_part, offset)[0]; offset += 1
                    data_len = struct.unpack_from('!Q', variable_part, offset)[0]; offset += 8
                    
                    # Receive image data
                    image_data = self.recv_exact(sock, data_len)
                    if not image_data:
                        break
                    
                    # Calculate latency
                    receive_time = time.time()
                    ros_time = ts_sec + ts_nsec / 1e9
                    latency = receive_time - ros_time
                    
                    # Create and publish ROS2 message
                    self.publish_image(
                        frame_id, ts_sec, ts_nsec, height, width,
                        encoding, step, is_bigendian, image_data,
                        latency
                    )
                    
            except socket.timeout:
                self.get_logger().warn("Socket timeout")
                self.reconnect(sock)
                sock = None
            except ConnectionError as e:
                self.get_logger().warn(f"Connection error: {e}")
                self.reconnect(sock)
                sock = None
            except Exception as e:
                self.get_logger().error(f"Unexpected error: {e}")
                self.reconnect(sock)
                sock = None
    
    def recv_exact(self, sock, n):
        """Receive exactly n bytes from socket"""
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)
    
    def publish_image(self, frame_id, ts_sec, ts_nsec, height, width,
                     encoding, step, is_bigendian, image_data, latency):
        """Publish received image to ROS2"""
        msg = Image()
        
        # Set header
        msg.header.stamp.sec = ts_sec
        msg.header.stamp.nanosec = ts_nsec
        msg.header.frame_id = f"frame_{frame_id}"
        
        # Set image properties
        msg.height = height
        msg.width = width
        msg.encoding = encoding
        msg.step = step
        msg.is_bigendian = is_bigendian
        msg.data = image_data
        
        # Publish
        self.publisher.publish(msg)
        
        # Update statistics
        self.frame_count += 1
        self.avg_latency = 0.9 * self.avg_latency + 0.1 * latency
        
    def reconnect(self, sock):
        """Handle reconnection with delay"""
        if sock:
            try:
                sock.close()
            except:
                pass
        time.sleep(self.reconnect_delay)
    
    def destroy_node(self):
        """Clean shutdown"""
        self.get_logger().info("ROS2 Image Client shutting down")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImageBridgeNode(host='127.0.0.1', port=21121)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()