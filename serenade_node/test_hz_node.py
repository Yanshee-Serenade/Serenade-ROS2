#!/usr/bin/env python3
"""
Fast Hz measurement script - bypasses all CLI overhead
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import time
import statistics
import sys

class HzTester(Node):
    def __init__(self, topic_name):
        super().__init__('hz_tester')
        
        # Statistics
        self.timestamps = []
        self.start_time = None
        self.count = 0
        self.last_print_time = time.time()
        self.print_interval = 2.0  # Print every 2 seconds
        
        # Fast subscription - minimal QoS
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        
        self.sub = self.create_subscription(
            Image,
            topic_name,
            self.callback,
            qos
        )
        
        self.get_logger().info(f"Testing Hz on {topic_name}")
        self.get_logger().info("Press Ctrl+C to stop")
    
    def callback(self, msg):
        now = time.time()
        
        if self.start_time is None:
            self.start_time = now
        
        # Store timestamp
        self.timestamps.append(now)
        self.count += 1
        
        # Keep only recent timestamps (last 5 seconds)
        cutoff = now - 5.0
        self.timestamps = [t for t in self.timestamps if t > cutoff]
        
        # Print statistics every print_interval seconds
        if now - self.last_print_time >= self.print_interval:
            if len(self.timestamps) >= 2:
                intervals = []
                for i in range(1, len(self.timestamps)):
                    intervals.append(self.timestamps[i] - self.timestamps[i-1])
                
                avg_interval = statistics.mean(intervals) if intervals else 0
                hz = 1.0 / avg_interval if avg_interval > 0 else 0
                
                min_interval = min(intervals) if intervals else 0
                max_interval = max(intervals) if intervals else 0
                
                self.get_logger().info(
                    f"Hz: {hz:.3f} | "
                    f"Avg interval: {avg_interval*1000:.1f}ms | "
                    f"Min: {min_interval*1000:.1f}ms | "
                    f"Max: {max_interval*1000:.1f}ms | "
                    f"Count: {self.count}"
                )
            else:
                self.get_logger().info(f"Waiting for messages... Count: {self.count}")
            
            self.last_print_time = now

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_hz_node.py <topic_name>")
        print("Example: python3 test_hz_node.py /camera/image_slow")
        return
    
    topic_name = sys.argv[1]
    
    rclpy.init()
    tester = HzTester(topic_name)
    
    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        pass
    finally:
        # Final statistics
        if tester.count > 0 and tester.start_time:
            total_time = time.time() - tester.start_time
            avg_hz = tester.count / total_time
            
            tester.get_logger().info("\n=== FINAL STATISTICS ===")
            tester.get_logger().info(f"Total messages: {tester.count}")
            tester.get_logger().info(f"Total time: {total_time:.2f}s")
            tester.get_logger().info(f"Average Hz: {avg_hz:.3f}")
            tester.get_logger().info(f"Expected Hz: ~5.0")
        
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
