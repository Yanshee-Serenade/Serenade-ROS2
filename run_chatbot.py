"""
Chatbot voice assistant - main entry point.

This file provides backward compatibility and serves as the main entry point
for the modular chatbot system. It imports and uses the modular chatbot package.
"""

import asyncio

import YanAPI
from chatbot.main import main

# 初始化机器人连接
YanAPI.yan_api_init("raspberrypi")

if __name__ == "__main__":
    # 运行助手
    asyncio.run(main())
