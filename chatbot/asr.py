"""
Speech recognition module for voice assistant.

This module provides the VoiceASR class for handling speech recognition
with continuous monitoring and callback support.
"""

import asyncio
import threading
import time
from typing import Any, Callable, Coroutine, Optional

import YanAPI


class VoiceASR:
    """语音识别封装"""

    def __init__(self):
        self.current_text = ""
        self.is_running = False
        self.callback: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None
        self._thread: Optional[threading.Thread] = None
        self._loop = asyncio.get_event_loop()  # 获取事件循环

    def start(self, callback: Callable[[str], Coroutine[Any, Any, None]]):
        """开始语音识别"""
        self.callback = callback
        self.is_running = True
        self.current_text = ""

        # 停止之前的ASR
        YanAPI.stop_voice_asr()
        timestamp = int(time.time())

        # 启动ASR
        YanAPI.start_voice_asr(continues=True, timestamp=timestamp)

        # 启动监控线程
        self._thread = threading.Thread(target=self._monitor_loop, args=(timestamp,))
        self._thread.daemon = True
        self._thread.start()

    def _monitor_loop(self, timestamp: int):
        """监控ASR状态"""
        while self.is_running:
            try:
                state = YanAPI.get_voice_asr_state()

                if state.get("code") == 0 and state.get("timestamp") == timestamp:
                    data = state.get("data", {})
                    intent = data.get("intent", {})
                    new_text = intent.get("text", "")

                    # 如果有新文本且不同于当前文本，调用回调
                    if new_text and new_text != self.current_text:
                        self.current_text = new_text
                        if self.callback:
                            # 线程安全地调度异步回调
                            asyncio.run_coroutine_threadsafe(
                                self.callback(new_text), self._loop
                            )

                    # 如果状态变为idle，停止监控
                    if state.get("status") == "idle":
                        self.is_running = False
                        break

                time.sleep(0.1)

            except Exception as e:
                print(f"ASR监控错误: {e}")
                self.is_running = False
                break

    def stop(self):
        """停止语音识别"""
        self.is_running = False
        YanAPI.stop_voice_asr()
