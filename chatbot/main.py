#!/usr/bin/env python3
"""
Main entry point for the chatbot voice assistant with ROS2 integration.

This module provides the main() function to run the voice assistant
and contains the SEGMENT_TTS configuration constant.
"""

import asyncio

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .assistant import VoiceAssistant

# 不启用分段 TTS，因为 TTS 分段延迟太大。在模型 8bit 量化的情况下，鸡煲输出的短一些延迟就会很低
SEGMENT_TTS = False


class ChatbotNode(Node):
    """ROS2 Node for the voice chatbot"""
    
    def __init__(self):
        super().__init__('chatbot_node')
        self.assistant = VoiceAssistant(segment_tts=SEGMENT_TTS)
        self.assistant.initialize_ros2(self)
        self.assistant.start()
        self.get_logger().info("Chatbot node started")


async def main():
    """主函数 - 极简使用方式"""
    rclpy.init()
    
    node = ChatbotNode()
    
    try:
        # Keep the node spinning
        while True:
            rclpy.spin_once(node, timeout_sec=0.1)
            await asyncio.sleep(0.01)
    except KeyboardInterrupt:
        print("\n👋 正在退出...")
        node.assistant.stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    # 运行助手
    asyncio.run(main())
