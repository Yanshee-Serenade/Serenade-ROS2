"""
Text-to-speech module for voice assistant.

This module provides the StreamTTS class for handling text-to-speech
with streaming and interrupt support.
"""

import asyncio
import time

import YanAPI


class StreamTTS:
    """流式TTS封装"""

    def __init__(self):
        self.queue = asyncio.Queue()
        self.is_running = False

    async def add_text(self, text: str, interrupt: bool = False):
        """添加文本到TTS队列"""
        await self.queue.put((text, interrupt))
        if not self.is_running:
            asyncio.create_task(self._process_queue())

    async def _process_queue(self):
        """处理TTS队列"""
        self.is_running = True

        while not self.queue.empty():
            text, interrupt = await self.queue.get()

            try:
                # 停止当前播放（如果可打断）
                if interrupt:
                    YanAPI.stop_voice_tts()

                # 开始新的语音合成
                timestamp = int(time.time())
                result = YanAPI.start_voice_tts(
                    tts=text, interrupt=interrupt, timestamp=timestamp
                )

                if result["code"] == 0:
                    # 等待播放完成
                    await self._wait_for_completion(timestamp)

            except Exception as e:
                print(f"TTS错误: {e}")
            finally:
                self.queue.task_done()

        self.is_running = False

    async def _wait_for_completion(self, timestamp: int):
        """等待TTS播放完成"""
        while True:
            try:
                status = YanAPI.get_voice_tts_state(timestamp)
                if status.get("status") == "idle":
                    break
                await asyncio.sleep(0.5)
            except Exception:
                break
