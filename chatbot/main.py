"""
Main entry point for the chatbot voice assistant.

This module provides the main() function to run the voice assistant
and contains the SEGMENT_TTS configuration constant.
"""

import asyncio

from .assistant import VoiceAssistant

# 不启用分段 TTS，因为 TTS 分段延迟太大。在模型 8bit 量化的情况下，鸡煲输出的短一些延迟就会很低
SEGMENT_TTS = False


async def main():
    """主函数 - 极简使用方式"""
    assistant = VoiceAssistant(segment_tts=SEGMENT_TTS)

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
