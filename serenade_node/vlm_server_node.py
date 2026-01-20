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


class VLMServerNode(Node):
    """ROS2 node for VLM inference"""

    def __init__(self):
        super().__init__('vlm_server_node')
        
        # Declare parameters
        self.declare_parameter('image_topic', '/camera/image_slow')
        self.declare_parameter('model_name', 'Qwen/Qwen3-VL-8B-Instruct')
        self.declare_parameter('max_new_tokens', 256)
        
        # Get parameter values
        self.image_topic = self.get_parameter('image_topic').value
        self.model_name = self.get_parameter('model_name').value
        self.max_new_tokens = self.get_parameter('max_new_tokens').value

        assert self.image_topic is not None, "Image topic must be provided"
        assert self.model_name is not None, "Model name must be provided"
        assert self.max_new_tokens is not None, "Max new tokens must be provided"

        self.processor: Optional[Any] = None
        self.model_vlm: Optional[Any] = None
        self.cv_bridge = CvBridge()
        self.latest_image = None
        
        # 日志相关变量
        self.generation_start_time = None
        self.generation_first_token_time = None
        self.total_tokens_generated = 0
        
        # Create subscriber and publisher
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
        
        # Load model
        self.load_model(self.model_name)
        self.get_logger().info("VLM Server Node initialized")

    def load_model(self, model_path: str):
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
                device_map="cuda",
            )
            self.model_vlm.eval()
            self.get_logger().info("✅ VLM model loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"VLM model loading failed: {str(e)}")
            raise

    def on_image(self, msg: Image):
        """Handle incoming camera image"""
        try:
            # Store in BGR format directly for saving as JPG
            self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            print(f"Error processing image: {e}")
            self.latest_image = None

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
        # 重置日志变量
        self.generation_start_time = time.time()
        self.generation_first_token_time = None
        self.total_tokens_generated = 0
        
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
                "max_new_tokens": self.max_new_tokens,
                "do_sample": True,
                "num_beams": 1,
            }

            # Start generation thread
            generation_thread = Thread(
                target=lambda: model_vlm.generate(**generation_kwargs)
            )
            generation_thread.start()

            # Stream response as text chunks via ROS2 topic
            first_token_received = False
            for text in streamer:
                if text:
                    # 记录第一个token到达的时间
                    if not first_token_received:
                        self.generation_first_token_time = time.time()
                        start_duration = self.generation_first_token_time - self.generation_start_time
                        self.get_logger().info(f"⏱️ 启动时间: {start_duration:.2f}秒")
                        first_token_received = True
                    
                    # 计算token数量（简单估算）
                    token_count = len(text.split())
                    self.total_tokens_generated += token_count
                    
                    msg = String()
                    msg.data = text
                    self.answer_publisher.publish(msg)

            # 生成完成后计算统计信息
            generation_end_time = time.time()
            total_duration = generation_end_time - self.generation_start_time
            
            # 计算有效生成时间（从第一个token开始）
            if self.generation_first_token_time:
                effective_duration = generation_end_time - self.generation_first_token_time
                if effective_duration > 0 and self.total_tokens_generated > 0:
                    tokens_per_second = self.total_tokens_generated / effective_duration
                    self.get_logger().info(f"📊 生成统计: {self.total_tokens_generated} tokens, {total_duration:.2f}秒 (有效生成: {effective_duration:.2f}秒)")
                    self.get_logger().info(f"🚀 Token速度: {tokens_per_second:.2f} tokens/秒")
                else:
                    self.get_logger().info(f"📊 生成统计: {total_duration:.2f}秒 (无有效token生成)")
            else:
                self.get_logger().info(f"📊 生成统计: {total_duration:.2f}秒 (未收到token)")

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