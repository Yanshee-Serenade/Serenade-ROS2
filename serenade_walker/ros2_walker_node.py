#!/usr/bin/env python3
"""
Walker ROS2 Node - Runs the robot walker indefinitely.
"""

import time
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

from serenade_walker import create_walker, WalkStraightSequence
from serenade_walker.controller import RobotWalker


class WalkerNode(Node):
    """ROS2 Node for walker"""
    
    def __init__(self):
        super().__init__('walker_node')
        self.walker: Optional[RobotWalker] = None
        
        # Subscribe to camera pose topic
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            '/orb_slam3/camera_pose',
            self.pose_callback,
            10
        )
        
    def pose_callback(self, msg: PoseStamped):
        """Callback for camera pose updates"""
        if self.walker:
            self.walker.camera_pose = msg


def main(args=None):
    """Main function to run walker ROS2 node"""
    rclpy.init(args=args)
    
    # Create walker node
    node = WalkerNode()
    node.get_logger().info("Walker node starting...")
    
    # Initialize walker
    try:
        walker = create_walker(
            period_ms=400
        )
        node.walker = walker
        node.get_logger().info("Walker initialized successfully")
    except Exception as e:
        node.get_logger().error(f"Failed to initialize walker: {str(e)}")
        node.destroy_node()
        rclpy.shutdown()
        return
    
    # Run walking indefinitely
    try:
        node.get_logger().info("Starting infinite walking sequence...")
        walk_sequence = WalkStraightSequence(backward=False)
        
        while rclpy.ok():
            # Run the walking sequence repeatedly
            walker.run_sequence(walk_sequence)
            
            # Brief pause between sequences
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down walker...")
    except Exception as e:
        node.get_logger().error(f"Walker error: {str(e)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
