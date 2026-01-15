"""
LLM client module for voice assistant.

This module provides the LLMClient class for communicating with LLM APIs
with streaming response support. Supports all endpoints from the Flask server.
"""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import aiohttp
from aiohttp import ClientError


@dataclass
class HealthStatus:
    """Health check response data."""

    status: str
    timestamp: str
    models: Dict[str, bool]


@dataclass
class ModelInfo:
    """Model information."""

    loaded: bool
    default: str
    available: List[str]


@dataclass
class DepthResponse:
    """Depth generation response data."""

    timestamp: str
    image_path: str
    depth_map_shape: tuple
    depth_plot_path: str
    da3_depth_path: str
    da3_depth_keypoints_path: str


class LLMClientError(Exception):
    """Base exception for LLM client errors."""

    pass


class LLMClient:
    """LLM客户端封装，支持所有API端点"""

    def __init__(self, base_url: str = "http://localhost:21122"):
        """
        Initialize LLM client.

        Args:
            base_url: Base URL of the Flask server (default: http://localhost:21122)
        """
        self.base_url = base_url.rstrip("/")
        self.generate_url = f"{self.base_url}/generate"
        self.depth_url = f"{self.base_url}/depth"
        self.health_url = f"{self.base_url}/health"
        self.models_url = f"{self.base_url}/models"
        self.timeout = aiohttp.ClientTimeout(
            total=300
        )  # 5 minutes timeout for long operations

    async def query_stream(
        self,
        text: str,
        callback: Callable[[str], None],
        segment_tts: bool = False,
        system_prompt: Optional[str] = None,
    ) -> None:
        """
        流式查询LLM生成文本。

        Args:
            text: 用户输入的文本
            callback: 回调函数，接收生成的文本片段
            segment_tts: 是否启用分段TTS（按句子分割）
            system_prompt: 自定义系统提示词，如果为None则使用默认提示词

        Raises:
            LLMClientError: 如果请求失败
        """
        if system_prompt is None:
            system_prompt = (
                "你是一个人形机器人，你叫鸡煲。图片是你看到的场景，"
                "请回答 <question> 标签中的用户问题，要求尽可能简短回答！"
            )

        payload = {"text": f"{system_prompt}<question>{text}</question>"}
        buffer = ""

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(self.generate_url, json=payload) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        raise LLMClientError(
                            f"Server error {response.status}: {error_data.get('error', 'Unknown error')}"
                        )

                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                text_chunk = data.get("text", "")

                                if text_chunk:
                                    buffer += text_chunk

                                    # 检查是否有完整句子可以输出
                                    if segment_tts:
                                        match = re.search(r"[，。？！；]", buffer)
                                        if match:
                                            pos = match.start()
                                            sentence = buffer[: pos + 1]
                                            callback(sentence)  # 通知完整句子
                                            buffer = buffer[pos + 1 :].lstrip()
                                    else:
                                        # 如果不启用分段TTS，直接回调每个chunk
                                        callback(text_chunk)

                            except json.JSONDecodeError:
                                # 忽略无效的JSON数据
                                pass

            # 输出剩余内容
            if buffer:
                if segment_tts:
                    callback(buffer)
                else:
                    # 如果没有分段TTS，确保所有内容都被回调
                    callback(buffer)

        except ClientError as e:
            raise LLMClientError(f"Network error: {str(e)}")
        except asyncio.TimeoutError:
            raise LLMClientError("Request timeout after 5 minutes")

    async def query(self, text: str, system_prompt: Optional[str] = None) -> str:
        """
        非流式查询LLM生成文本（收集所有响应后返回）。

        Args:
            text: 用户输入的文本
            system_prompt: 自定义系统提示词

        Returns:
            完整的生成文本

        Raises:
            LLMClientError: 如果请求失败
        """
        result = []

        def callback(chunk: str) -> None:
            result.append(chunk)

        await self.query_stream(
            text, callback, segment_tts=False, system_prompt=system_prompt
        )
        return "".join(result)

    async def generate_depth(self) -> DepthResponse:
        """
        生成深度图。

        Returns:
            DepthResponse: 深度生成结果

        Raises:
            LLMClientError: 如果请求失败
        """
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(self.depth_url) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        raise LLMClientError(
                            f"Server error {response.status}: {error_data.get('error', 'Unknown error')}"
                        )

                    data = await response.json()

                    # 解析深度图形状
                    depth_shape_str = data.get("depth_map_shape", "")
                    if depth_shape_str:
                        # 假设形状格式为 "(height, width)" 或类似
                        depth_shape = (
                            eval(depth_shape_str)
                            if isinstance(depth_shape_str, str)
                            else tuple(depth_shape_str)
                        )
                    else:
                        depth_shape = (0, 0)

                    return DepthResponse(
                        timestamp=data.get("timestamp", ""),
                        image_path=data.get("image_path", ""),
                        depth_map_shape=depth_shape,
                        depth_plot_path=data.get("depth_plot_path", ""),
                        da3_depth_path=data.get("da3_depth_path", ""),
                        da3_depth_keypoints_path=data.get(
                            "da3_depth_keypoints_path", ""
                        ),
                    )

        except ClientError as e:
            raise LLMClientError(f"Network error: {str(e)}")
        except asyncio.TimeoutError:
            raise LLMClientError("Request timeout after 5 minutes")
        except json.JSONDecodeError as e:
            raise LLMClientError(f"Invalid JSON response: {str(e)}")

    async def check_health(self) -> HealthStatus:
        """
        检查服务器健康状态。

        Returns:
            HealthStatus: 健康状态信息

        Raises:
            LLMClientError: 如果请求失败
        """
        try:
            async with aiohttp.ClientSession(timeout=10) as session:
                async with session.get(self.health_url) as response:
                    if response.status != 200:
                        raise LLMClientError(
                            f"Server returned status {response.status}"
                        )

                    data = await response.json()

                    models_data = data.get("models", {})
                    models_status = {
                        "vlm_loaded": models_data.get("vlm_loaded", False),
                        "da3_loaded": models_data.get("da3_loaded", False),
                        "sam3_loaded": models_data.get("sam3_loaded", False),
                    }

                    return HealthStatus(
                        status=data.get("status", "unknown"),
                        timestamp=data.get("timestamp", ""),
                        models=models_status,
                    )

        except ClientError as e:
            raise LLMClientError(f"Network error: {str(e)}")
        except asyncio.TimeoutError:
            raise LLMClientError("Health check timeout after 10 seconds")
        except json.JSONDecodeError as e:
            raise LLMClientError(f"Invalid JSON response: {str(e)}")

    async def list_models(self) -> Dict[str, ModelInfo]:
        """
        列出可用模型及其状态。

        Returns:
            Dict[str, ModelInfo]: 模型信息字典

        Raises:
            LLMClientError: 如果请求失败
        """
        try:
            async with aiohttp.ClientSession(timeout=10) as session:
                async with session.get(self.models_url) as response:
                    if response.status != 200:
                        raise LLMClientError(
                            f"Server returned status {response.status}"
                        )

                    data = await response.json()

                    models_info = {}
                    for model_type, model_data in data.items():
                        models_info[model_type] = ModelInfo(
                            loaded=model_data.get("loaded", False),
                            default=model_data.get("default", ""),
                            available=model_data.get("available", []),
                        )

                    return models_info

        except ClientError as e:
            raise LLMClientError(f"Network error: {str(e)}")
        except asyncio.TimeoutError:
            raise LLMClientError("Request timeout after 10 seconds")
        except json.JSONDecodeError as e:
            raise LLMClientError(f"Invalid JSON response: {str(e)}")

    async def is_server_healthy(self) -> bool:
        """
        检查服务器是否健康（简化版本）。

        Returns:
            bool: 服务器是否健康
        """
        try:
            health_status = await self.check_health()
            return health_status.status == "healthy"
        except LLMClientError:
            return False

    async def wait_for_server(
        self, max_retries: int = 30, retry_delay: float = 2.0
    ) -> bool:
        """
        等待服务器启动。

        Args:
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）

        Returns:
            bool: 服务器是否成功启动
        """
        for attempt in range(max_retries):
            try:
                if await self.is_server_healthy():
                    print(f"Server is ready after {attempt + 1} attempts")
                    return True
                else:
                    print(f"Attempt {attempt + 1}/{max_retries}: Server not ready yet")
            except LLMClientError as e:
                print(f"Attempt {attempt + 1}/{max_retries}: {str(e)}")

            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)

        print(f"Server failed to start after {max_retries} attempts")
        return False


# 同步包装器（可选，用于非异步环境）
class SyncLLMClient:
    """同步版本的LLM客户端"""

    def __init__(self, base_url: str = "http://localhost:21122"):
        self.client = LLMClient(base_url)
        self.loop = asyncio.new_event_loop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.loop.close()

    def query_stream(
        self,
        text: str,
        callback: Callable[[str], None],
        segment_tts: bool = False,
        system_prompt: Optional[str] = None,
    ) -> None:
        """同步流式查询"""
        return self.loop.run_until_complete(
            self.client.query_stream(text, callback, segment_tts, system_prompt)
        )

    def query(self, text: str, system_prompt: Optional[str] = None) -> str:
        """同步非流式查询"""
        return self.loop.run_until_complete(self.client.query(text, system_prompt))

    def generate_depth(self) -> DepthResponse:
        """同步生成深度图"""
        return self.loop.run_until_complete(self.client.generate_depth())

    def check_health(self) -> HealthStatus:
        """同步检查健康状态"""
        return self.loop.run_until_complete(self.client.check_health())

    def list_models(self) -> Dict[str, ModelInfo]:
        """同步列出模型"""
        return self.loop.run_until_complete(self.client.list_models())

    def is_server_healthy(self) -> bool:
        """同步检查服务器健康"""
        return self.loop.run_until_complete(self.client.is_server_healthy())

    def wait_for_server(self, max_retries: int = 30, retry_delay: float = 2.0) -> bool:
        """同步等待服务器启动"""
        return self.loop.run_until_complete(
            self.client.wait_for_server(max_retries, retry_delay)
        )
