"""
Voice assistant module for integrating ASR, LLM, and TTS with ROS2.

This module provides the VoiceAssistant class which integrates
speech recognition, LLM querying, and text-to-speech functionality
with ROS2 topic communication.
"""

import asyncio

import rclpy
from std_msgs.msg import String

from .asr import VoiceASR
from .tts import StreamTTS


class VoiceAssistant:
    """语音助手 - 集成ASR、LLM、TTS"""

    def __init__(self, segment_tts: bool = False):
        """
        Initialize voice assistant.

        Args:
            segment_tts: Whether to enable segmented TTS (sentence-based)
        """
        self.asr = VoiceASR()
        self.tts = StreamTTS()
        self.segment_tts = segment_tts
        self.is_running = False
        self.current_response = ""
        
        # ROS2 node and publishers/subscribers
        self.node = None
        self.question_publisher = None
        self.answer_subscriber = None

    def initialize_ros2(self, node):
        """Initialize ROS2 node and publishers/subscribers"""
        self.node = node
        self.question_publisher = node.create_publisher(String, 'question', 10)
        self.answer_subscriber = node.create_subscription(
            String,
            'answer',
            self.on_answer,
            10
        )

    def on_answer(self, msg: String):
        """Handle incoming answers from VLM server via ROS2 topic"""
        text_chunk = msg.data
        if text_chunk:
            self.current_response += text_chunk
            print(f"🤖 AI: {text_chunk}", flush=True)

            # 将响应添加到TTS队列
            asyncio.create_task(self.tts.add_text(text_chunk, interrupt=False))

    def start(self):
        """启动语音助手"""
        self.is_running = True
        print("🎤 语音助手已启动，开始说话吧...")

        # 启动语音识别
        self.asr.start(self._on_speech_recognized)

    def stop(self):
        """停止语音助手"""
        self.is_running = False
        self.asr.stop()
        print("🛑 语音助手已停止")

    async def _on_speech_recognized(self, text: str):
        """当语音被识别时的回调"""
        print(f"🗣️ 你说: {text}")
        if "问你" not in text:
            print("👀 如果提问，请以“问你”开头。")
            return

        # 异步处理LLM查询
        await self._process_query(text)

    async def _process_query(self, text: str):
        """处理查询并生成响应"""
        print("🤔 思考中...")
        self.current_response = ""

        # 发布问题到ROS2主题
        if self.question_publisher:
            msg = String()
            msg.data = text
            self.question_publisher.publish(msg)
