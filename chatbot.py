import asyncio
import json
import re
import threading
import time
from typing import Any, Callable, Coroutine, Optional

import aiohttp

import YanAPI

# 初始化机器人连接
YanAPI.yan_api_init("raspberrypi")

# 不启用分段 TTS，因为 TTS 分段延迟太大。在模型 8bit 量化的情况下，鸡煲输出的短一些延迟就会很低
SEGMENT_TTS = False


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


class LLMClient:
    """LLM客户端封装"""

    def __init__(self, url: str = "http://10.249.8.158:51122/generate"):
        self.url = url

    async def query_stream(self, text: str, callback: Callable[[str], None]):
        """流式查询LLM"""
        payload = {
            "text": f"你是一个人形机器人，你叫鸡煲。图片是你看到的场景，请回答 <question> 标签中的用户问题，要求尽可能简短回答！<question>{text}</question>"
        }
        buffer = ""

        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json=payload) as response:
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            text_chunk = data.get("text", "")

                            if text_chunk:
                                buffer += text_chunk

                                # 检查是否有完整句子可以输出
                                match = re.search(r"[，。？！；]", buffer)
                                if match and SEGMENT_TTS:
                                    pos = match.start()
                                    sentence = buffer[: pos + 1]
                                    callback(sentence)  # 通知完整句子
                                    buffer = buffer[pos + 1 :].lstrip()
                                    match = re.search(r"[，。？！；]", buffer)

                        except json.JSONDecodeError:
                            pass

        # 输出剩余内容
        if buffer:
            callback(buffer)


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


class VoiceAssistant:
    """语音助手 - 集成ASR、LLM、TTS"""

    def __init__(self):
        self.asr = VoiceASR()
        self.llm = LLMClient()
        self.tts = StreamTTS()
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
        await self.llm.query_stream(text, self._on_llm_response)

    def _on_llm_response(self, text_chunk: str):
        """当收到LLM响应时的回调"""
        if text_chunk:
            self.current_response += text_chunk
            print(f"🤖 AI: {text_chunk}", flush=True)

            # 将响应添加到TTS队列
            asyncio.create_task(self.tts.add_text(text_chunk, interrupt=False))


async def main():
    """主函数 - 极简使用方式"""
    assistant = VoiceAssistant()

    try:
        # 启动助手
        assistant.start()

        # 保持运行
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n👋 正在退出...")
        assistant.stop()


if __name__ == "__main__":
    # 运行助手
    asyncio.run(main())
