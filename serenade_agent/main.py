#!/usr/bin/env python3
"""
Main VLM agent node for Serenade ROS2 robot.
Handles question processing, VLM inference, and command publishing.
"""

import asyncio
import queue
import time
from datetime import datetime
from threading import Thread, Event
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from serenade_agent.config import INTERRUPTIBLE_FLAG
from serenade_agent.vlm_operations import VLMOperations
from serenade_agent.message_builder import MessageBuilder
from serenade_agent.response_parser import ResponseParser


class InterruptedError(Exception):
    """Exception raised when the generation process is interrupted."""
    pass

class InterruptibleStreamer:
    """
    Wrapper around the VLM streamer to check for interruption signals
    during the iteration of generated tokens.
    """
    def __init__(self, streamer, interrupt_event: Event):
        self.streamer = streamer
        self.interrupt_event = interrupt_event
        self.iterator = None

    def __iter__(self):
        self.iterator = iter(self.streamer)
        return self

    def __next__(self):
        # Check for interruption before retrieving the next token
        if self.interrupt_event.is_set():
            raise InterruptedError("Generation interrupted by new incoming question")
        return next(self.iterator)

class AgentNode(Node):
    """ROS2 node for VLM agent with conversation history management."""

    def __init__(self):
        super().__init__('agent_node')

        # Declare parameters
        self.declare_parameter('image_topic', '/camera/image_raw')
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

        # Load model
        self.load_model()

        # Initialize question queue and worker thread
        self.question_queue: queue.Queue = queue.Queue()
        self.worker_running = True
        
        # Interruption control
        self.interrupt_event = Event()
        self.processing_interruptible = False

        self.worker_thread = Thread(target=self._question_worker, daemon=True)
        self.worker_thread.start()

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
        self.question_publisher = self.create_publisher(String, 'question', 10)
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

        # Node initialized
        self.get_logger().info("VLM Agent Node initialized")

    def load_model(self):
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
        """Handle incoming questions from ROS2 topic by queueing them."""
        question_text = msg.data
        is_interruptible = question_text.startswith(INTERRUPTIBLE_FLAG)
        
        if self.processing_interruptible:
            if is_interruptible:
                # If both interruptible, the latter is discarded
                self.get_logger().info(f"Ignoring question: {question_text}")
                return
            else:
                # If new question is not interruptible, interrupt the former
                self.get_logger().info("Interrupting current question for new incoming request.")
                self.interrupt_event.set()

        self.get_logger().info(f"Received question: {question_text}")
        # Add question to queue for sequential processing
        self.question_queue.put(question_text)

    async def _process_question(self, question_text: str):
        """Process the question and stream the answer."""
        # Reset metrics
        self.generation_start_time = time.time()
        self.generation_first_token_time = None
        self.total_tokens_generated = 0

        # Check for interruptible flag
        is_interruptible = question_text.startswith(INTERRUPTIBLE_FLAG)
        if is_interruptible:
            # Remove flag
            question_text = question_text[len(INTERRUPTIBLE_FLAG):]
            self.processing_interruptible = True
            self.interrupt_event.clear()
        else:
            self.processing_interruptible = False

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
            
            # If interruptible, wrap the streamer to catch the interrupt event
            # during the parsing loop
            if is_interruptible:
                streamer_to_use = InterruptibleStreamer(streamer, self.interrupt_event)
            else:
                streamer_to_use = streamer

            generation_kwargs = self.vlm_ops.get_generation_kwargs(inputs, streamer)

            # Start generation thread
            generation_thread = Thread(target=lambda: model.generate(**generation_kwargs))
            generation_thread.start()

            # Parse streamed response
            # If interrupted, InterruptedError will be raised from inside this call (via iterator)
            self.response_parser.parse_streamed_response(streamer_to_use)

            # If parse_streamed_response ends, we disable interruption to ensure 
            # we finish saving the history and don't interrupt the cleanup/saving phase.
            self.processing_interruptible = False

            # Get the parsed results for history saving
            say_content = self.response_parser.get_say()
            setstate_line = self.response_parser.get_setstate()

            # Add assistant response to history (save the full streamed output)
            full_response = f"say {say_content}\n{setstate_line}"
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

        except InterruptedError:
            self.get_logger().warn("⚠️ Question processing interrupted by new request.")
            # Remove the just-added user message so it doesn't pollute history
            if hasattr(self.message_builder, 'messages') and isinstance(self.message_builder.messages, list):
                if self.message_builder.messages:
                    self.message_builder.messages.pop()
            # Do NOT add assistant message or save history
            return

        except Exception as e:
            self.get_logger().error(f"VLM error: {str(e)}")
            # Publish error message
            error_msg = String()
            error_msg.data = f"VLM error: {str(e)}"
            self.answer_publisher.publish(error_msg)
            
        finally:
            # Always ensure interruptible state is reset
            self.processing_interruptible = False

    def _question_worker(self):
        """Worker thread that processes questions sequentially from the queue."""
        while self.worker_running:
            try:
                # Get question from queue with timeout to allow graceful shutdown
                question_text = self.question_queue.get(timeout=2.0)
                
                # Process question in async event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self._process_question(question_text))
                finally:
                    loop.close()
                    
                self.question_queue.task_done()
            except queue.Empty:
                # Publish a interruptible question for self-reasoning
                msg = String()
                msg.data = f"{INTERRUPTIBLE_FLAG}请简要概括现在的情况，并决定下一步行动"
                self.question_publisher.publish(msg)
                continue
            except Exception as e:
                self.get_logger().error(f"Error in question worker: {str(e)}")

def main(args=None):
    """Main function to run VLM agent ROS2 node."""
    rclpy.init(args=args)
    agent_node = AgentNode()

    try:
        rclpy.spin(agent_node)
    except KeyboardInterrupt:
        agent_node.get_logger().info("Shutting down VLM Agent...")
    finally:
        # Stop the worker thread gracefully
        agent_node.worker_running = False
        agent_node.worker_thread.join(timeout=5.0)
        
        agent_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
