#!/usr/bin/env python3
"""
VLM Server - ROS2 node for Vision Language Model inference.
Subscribes to 'question' topic and publishes answers to 'answer' topic.
"""

import datetime
import time
from typing import Any, Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from threading import Thread

from serenade_server.config import config


class VLMServerNode(Node):
    """ROS2 node for VLM inference"""

    def __init__(self):
        super().__init__('vlm_server_node')
        
        self.processor: Optional[Any] = None
        self.model_vlm: Optional[Any] = None
        self.cv_bridge = CvBridge()
        self.latest_image = None
        
        # Create subscriber and publisher
        self.question_subscription = self.create_subscription(
            String,
            'question',
            self.on_question,
            10
        )
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.on_image,
            1
        )
        self.answer_publisher = self.create_publisher(String, 'answer', 10)
        
        # Load model
        self.load_model()
        self.get_logger().info("VLM Server Node initialized")

    def load_model(self, model_path: str = config.MODEL_VLM_DEFAULT):
        """Load VLM model."""
        self.get_logger().info(f"Loading VLM model from {model_path}...")
        try:
            # Load processor
            self.get_logger().info(f"Loading model processor...")
            self.processor = AutoProcessor.from_pretrained(model_path)

            # Load model
            self.get_logger().info(f"Loading model weights...")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            self.model_vlm = AutoModelForImageTextToText.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
            )
            self.model_vlm.eval()
            self.get_logger().info("✅ VLM model loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"VLM model loading failed: {str(e)}")
            raise

    def on_image(self, msg: Image):
        """Handle incoming camera image"""
        try:
            self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {str(e)}")

    def on_question(self, msg: String):
        """Handle incoming questions from ROS2 topic"""
        question_text = msg.data
        self.get_logger().info(f"Received question: {question_text}")
        
        # Process question asynchronously to avoid blocking the main thread
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._process_question(question_text))
        except Exception as e:
            self.get_logger().error(f"Error processing question: {str(e)}")

    async def _process_question(self, question_text: str):
        """Process the question and stream the answer"""
        try:
            # Build messages for the VLM
            content = []
            
            # Add image if available
            if self.latest_image is not None:
                content.append({"type": "image", "image": self.latest_image})
            
            # Add text question
            content.append({"type": "text", "text": question_text})
            
            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]

            # Get model components
            if self.processor is None or self.model_vlm is None:
                raise ValueError("VLM model not loaded")
            processor = self.processor
            model_vlm = self.model_vlm

            # Prepare inputs
            if hasattr(processor, "apply_chat_template"):
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            else:
                tokenizer = (
                    processor.tokenizer
                    if hasattr(processor, "tokenizer")
                    else processor
                )
                text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                inputs = tokenizer(text, return_tensors="pt")

            # Move to device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            inputs = inputs.to(device=device, dtype=dtype)

            # Setup streamer
            tokenizer = (
                processor.tokenizer if hasattr(processor, "tokenizer") else processor
            )
            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            # Generation parameters
            generation_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": config.MAX_NEW_TOKENS,
                "do_sample": True,
                "num_beams": 1,
            }

            # Start generation thread
            generation_thread = Thread(
                target=lambda: model_vlm.generate(**generation_kwargs)
            )
            generation_thread.start()

            # Stream response as text chunks via ROS2 topic
            for text in streamer:
                if text:
                    msg = String()
                    msg.data = text
                    self.answer_publisher.publish(msg)

            self.get_logger().info("✅ VLM generation completed")

        except Exception as e:
            self.get_logger().error(f"VLM error: {str(e)}")
            # Publish error message
            error_msg = String()
            error_msg.data = f"VLM error: {str(e)}"
            self.answer_publisher.publish(error_msg)


def main(args=None):
    """Main function to run VLM ROS2 server"""
    rclpy.init(args=args)
    
    vlm_node = VLMServerNode()
    
    try:
        rclpy.spin(vlm_node)
    except KeyboardInterrupt:
        vlm_node.get_logger().info("Shutting down VLM Server...")
    finally:
        vlm_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
