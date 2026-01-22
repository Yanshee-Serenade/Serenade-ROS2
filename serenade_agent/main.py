#!/usr/bin/env python3
"""
Main VLM agent node for Serenade ROS2 robot.
Handles question processing, VLM inference, and command publishing.
"""

import asyncio
import time
from datetime import datetime
from threading import Thread
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from serenade_agent.vlm_operations import VLMOperations
from serenade_agent.message_builder import MessageBuilder
from serenade_agent.response_parser import ResponseParser


class AgentNode(Node):
    """ROS2 node for VLM agent with conversation history management."""

    def __init__(self):
        super().__init__('agent_node')

        # Declare parameters
        self.declare_parameter('image_topic', '/camera/image_slow')
        self.declare_parameter('model_name', 'Qwen/Qwen3-VL-8B-Instruct')
        self.declare_parameter('max_new_tokens', 256)
        self.declare_parameter('use_history', False)

        # Get parameter values
        self.image_topic = self.get_parameter('image_topic').value
        self.model_name = self.get_parameter('model_name').value
        self.max_new_tokens = self.get_parameter('max_new_tokens').value
        self.use_history = self.get_parameter('use_history').value

        assert self.image_topic is not None, "Image topic must be provided"
        assert self.model_name is not None, "Model name must be provided"
        assert self.max_new_tokens is not None, "Max new tokens must be provided"
        assert self.use_history is not None, "Use history flag must be provided"

        self.cv_bridge = CvBridge()
        self.latest_image: Optional[np.ndarray] = None
        self.latest_image_path: Optional[str] = None

        # Initialize VLM operations and message builder
        self.vlm_ops = VLMOperations(self.model_name, self.max_new_tokens)
        self.message_builder = MessageBuilder(use_history=self.use_history)

        # Create subscribers and publishers first
        self.question_subscription = self.create_subscription(
            String,
            'question',
            self.on_question,
            10
        )
        self.image_subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.on_image,
            1
        )
        self.answer_publisher = self.create_publisher(String, 'answer', 10)
        self.target_publisher = self.create_publisher(String, 'target', 10)

        # Initialize response parser with dependencies
        self.response_parser = ResponseParser(
            logger=self.get_logger(),
            answer_publisher=self.answer_publisher,
            target_publisher=self.target_publisher,
            message_builder=self.message_builder
        )

        # Timing metrics
        self.generation_start_time: Optional[float] = None
        self.generation_first_token_time: Optional[float] = None
        self.total_tokens_generated = 0

        # Load model
        self.load_model(self.model_name)
        self.get_logger().info("VLM Agent Node initialized")

    def load_model(self, model_path: str):
        """Load VLM model."""
        try:
            self.vlm_ops.load_model(logger=self.get_logger())
        except Exception as e:
            self.get_logger().error(f"Failed to load VLM model: {str(e)}")
            raise

    def on_image(self, msg: Image):
        """Handle incoming camera image."""
        try:
            self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
            self.latest_image = None

    def on_question(self, msg: String):
        """Handle incoming questions from ROS2 topic."""
        question_text = msg.data
        self.get_logger().info(f"Received question: {question_text}")

        # Process question asynchronously to avoid blocking the main thread
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._process_question(question_text))
        except Exception as e:
            self.get_logger().error(f"Error processing question: {str(e)}")

    async def _process_question(self, question_text: str):
        """Process the question and stream the answer."""
        # Reset metrics
        self.generation_start_time = time.time()
        self.generation_first_token_time = None
        self.total_tokens_generated = 0

        try:
            # Save image if available
            if self.latest_image is not None:
                self.latest_image_path = self.message_builder.save_image(self.latest_image)
            else:
                raise ValueError("Must have image")

            # Add user message to history
            self.message_builder.add_user_message(question_text, self.latest_image_path)

            # Get messages for VLM
            messages = self.message_builder.get_messages()

            # Prepare inputs
            inputs = self.vlm_ops.prepare_inputs(messages)
            model = self.vlm_ops.get_model()

            # Setup streamer
            streamer = self.vlm_ops.get_streamer()
            generation_kwargs = self.vlm_ops.get_generation_kwargs(inputs, streamer)

            # Start generation thread
            generation_thread = Thread(target=lambda: model.generate(**generation_kwargs))
            generation_thread.start()

            # Parse streamed response - the parser handles publishing and logging
            self.response_parser.parse_streamed_response(streamer)

            # Get the parsed results for history saving
            think_content = self.response_parser.get_think()
            say_content = self.response_parser.get_say()
            setstate_line = self.response_parser.get_setstate()

            # Add assistant response to history (save the full streamed output)
            full_response = f"think {think_content}\nsay {say_content}\n{setstate_line}"
            self.message_builder.add_assistant_message(full_response)

            # Save history with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            self.message_builder.save_history(timestamp)

            # Log metrics
            generation_end_time = time.time()
            total_duration = generation_end_time - self.generation_start_time

            if self.generation_first_token_time:
                effective_duration = generation_end_time - self.generation_first_token_time
                self.get_logger().info(
                    f"📊 Generated in {total_duration:.2f}s "
                    f"(effective: {effective_duration:.2f}s)"
                )
            else:
                self.get_logger().info(f"📊 Generated in {total_duration:.2f}s")

            self.get_logger().info("✅ VLM generation completed")

        except Exception as e:
            self.get_logger().error(f"VLM error: {str(e)}")
            # Publish error message
            error_msg = String()
            error_msg.data = f"VLM error: {str(e)}"
            self.answer_publisher.publish(error_msg)


def main(args=None):
    """Main function to run VLM agent ROS2 node."""
    rclpy.init(args=args)

    agent_node = AgentNode()

    try:
        rclpy.spin(agent_node)
    except KeyboardInterrupt:
        agent_node.get_logger().info("Shutting down VLM Agent...")
    finally:
        agent_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
