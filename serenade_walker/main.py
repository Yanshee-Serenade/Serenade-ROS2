#!/usr/bin/env python3
"""
Walker ROS2 Node - Runs the robot walker indefinitely.
"""

import time
import threading  # <--- IMPORT THREADING
from typing import Optional

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray

from serenade_walker.factory import create_walker
from serenade_walker.sequences import WalkTargetSequence
from serenade_walker.controller import RobotWalker


class WalkerNode(Node):
    """ROS2 Node for walker"""
    
    def __init__(self):
        super().__init__('walker_node')
        self.walker: Optional[RobotWalker] = None

        # UPDATED: Use SensorDataQoS or Best Effort for vision topics 
        # (Safer if publishers are Best Effort)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Subscribe to camera pose topic
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            '/orb_slam3/camera_pose',
            self.pose_callback,
            qos_profile
        )

        # Subscribe to marker topic
        self.marker_subscription = self.create_subscription(
            MarkerArray,
            '/yolo_world/markers',
            self.marker_callback,
            qos_profile
        )

        # Subscribe to target topic
        self.marker_subscription = self.create_subscription(
            String,
            '/target',
            self.target_callback,
            qos_profile
        )

        # Publisher for target lost questions
        self.question_publisher = self.create_publisher(String, '/question', 10)

    def pose_callback(self, msg: PoseStamped):
        if self.walker:
            self.walker.camera_pose = msg
    
    def marker_callback(self, msg: MarkerArray):
        if self.walker:
            self.walker.marker_array = msg
    
    def target_callback(self, msg: String):
        self.get_logger().info(f"Target changed to {msg.data}")
        if self.walker:
            self.walker.target = msg.data


def main(args=None):
    """Main function to run walker ROS2 node"""
    rclpy.init(args=args)
    
    # Create walker node
    node = WalkerNode()
    node.get_logger().info("Walker node starting...")
    
    # Initialize walker
    try:
        walker = create_walker(
            period_ms=400,
            node=node
        )
        node.walker = walker
        node.get_logger().info("Walker initialized successfully")
    except Exception as e:
        node.get_logger().error(f"Failed to initialize walker: {str(e)}")
        node.destroy_node()
        rclpy.shutdown()
        return

    # --- THE FIX: Spin ROS in a background thread ---
    # This allows callbacks to update self.walker.camera_pose 
    # WHILE the while loop below is running.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    # -----------------------------------------------
    
    # Run walking indefinitely in the MAIN thread
    try:
        node.get_logger().info("Starting infinite walking sequence...")
        walk_sequence = WalkTargetSequence(backward=False)
        
        while rclpy.ok():
            # Run the walking sequence repeatedly
            # The callbacks in the background thread will keep updating 'walker' data
            walker.run_sequence(walk_sequence)
            
            # Brief pause between sequences
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down walker...")
    except Exception as e:
        node.get_logger().error(f"Walker error: {str(e)}")
    finally:
        # Clean up
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join()


if __name__ == '__main__':
    main()