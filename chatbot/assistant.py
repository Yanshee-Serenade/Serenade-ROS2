"""
Voice assistant module for integrating ASR, LLM, and TTS.

This module provides the VoiceAssistant class which integrates
speech recognition, LLM querying, and text-to-speech functionality.
"""

import asyncio

from .asr import VoiceASR
from .llm_client import LLMClient
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
        self.llm = LLMClient()
        self.tts = StreamTTS()
        self.segment_tts = segment_tts
        self.is_running = False
        self.current_response = ""

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

        # 流式查询LLM
        await self.llm.query_stream(
            text, self._on_llm_response, segment_tts=self.segment_tts
        )

    def _on_llm_response(self, text_chunk: str):
        """当收到LLM响应时的回调"""
        if text_chunk:
            self.current_response += text_chunk
            print(f"🤖 AI: {text_chunk}", flush=True)

            # 将响应添加到TTS队列
            asyncio.create_task(self.tts.add_text(text_chunk, interrupt=False))
