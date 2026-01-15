"""
VLM Client - Client for Vision Language Model TCP server.
Connects to VLM server on port 21122 using binary protocol.
Falls back to Doubao API if local server is unavailable.
"""

import asyncio
import os
import socket
import struct
from typing import Generator, Optional

from dotenv import load_dotenv

# --- VolcEngine Imports (Assumed available per requirements) ---
from volcenginesdkarkruntime import AsyncArk
from volcenginesdkarkruntime.types.responses import (
    ResponseCompletedEvent,
    ResponseOutputItemAddedEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)

from .config import client_config

# Load environment variables from .env file
load_dotenv()

# Doubao Configuration
DOUBAO_MODEL_ENDPOINT = "doubao-seed-1-6-251015"
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


class VLMClient:
    """Client for VLM TCP server with binary protocol and Doubao API fallback."""

    def __init__(
        self,
        host: str = client_config.VLM_HOST,
        port: int = client_config.VLM_PORT,
        timeout: float = client_config.RECV_TIMEOUT,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket: Optional[socket.socket] = None

    def connect(self):
        """Connect to VLM server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(self.timeout)
        self.socket.connect((self.host, self.port))

    def recv_all(self, length: int) -> bytes:
        """Receive exact number of bytes."""
        if self.socket is None:
            raise ConnectionError("Socket not connected")
        data = b""
        while len(data) < length:
            chunk = self.socket.recv(min(length - len(data), 4096))
            if not chunk:
                raise ConnectionError("Connection closed unexpectedly")
            data += chunk
        return data

    async def _doubao_stream_async(self, image_path: str, text_query: str):
        """
        Async generator that interacts with Doubao API using streaming.
        """
        api_key = os.getenv("ARK_API_KEY")
        if not api_key:
            raise ValueError("Environment variable 'ARK_API_KEY' is missing.")

        client = AsyncArk(base_url=DOUBAO_BASE_URL, api_key=api_key)

        abs_path = os.path.abspath(image_path)

        stream = await client.responses.create(
            model=DOUBAO_MODEL_ENDPOINT,
            input=[
                {  # pyright: ignore[reportArgumentType]
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": f"file://{abs_path}"},
                        {"type": "input_text", "text": text_query},
                    ],
                },
            ],
            caching={
                "type": "enabled",
            },
            store=True,
            stream=True,
        )

        async for event in stream:  # pyright: ignore[reportGeneralTypeIssues]
            # Handle specific event types as requested
            if isinstance(event, ResponseReasoningSummaryTextDeltaEvent):
                yield event.delta
            elif isinstance(event, ResponseTextDeltaEvent):
                yield event.delta
            # Optional: You can choose to yield or log other events if needed
            # For a pure text generator replacement, we usually just want the content deltas.
            # However, for debugging purposes (like in your snippet), here are the others:
            elif isinstance(event, ResponseOutputItemAddedEvent):
                # Usually metadata, skipping for pure text generation stream
                pass
            elif isinstance(event, ResponseTextDoneEvent):
                # End of specific block
                pass
            elif isinstance(event, ResponseCompletedEvent):
                # Final usage stats
                pass

    def fallback_to_doubao(
        self, image_path: str, text_query: str
    ) -> Generator[str, None, None]:
        """
        Wraps the async Doubao stream in a synchronous generator.
        """
        print("(⚠️ Local VLM server unavailable, falling back to Doubao API...)")

        try:
            # We need to bridge async generator to sync generator.
            # Ideally, one would rewrite the whole client to be async, but to fit the existing interface:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async_gen = self._doubao_stream_async(image_path, text_query)

            while True:
                try:
                    # Get the next chunk from the async generator
                    chunk = loop.run_until_complete(async_gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break

            loop.close()

        except ValueError as e:
            yield f"\nConfiguration Error: {str(e)}"
        except Exception as e:
            yield f"\nDoubao API Error: {str(e)}"

    def generate(self, image_path: str, text_query: str) -> Generator[str, None, None]:
        """
        Generate text from VLM with streaming response.
        Prioritizes Local TCP -> Falls back to Doubao API.
        """
        local_available = False

        # 1. Try to Connect Locally
        try:
            self.connect()
            local_available = True
        except (ConnectionRefusedError, socket.timeout, OSError):
            local_available = False

        # 2. Fallback Logic
        if not local_available:
            yield from self.fallback_to_doubao(image_path, text_query)
            return

        # 3. Local VLM Protocol Logic
        try:
            # Send image_path
            if self.socket is None:
                raise ConnectionError("Socket not connected")
            img_path_bytes = image_path.encode("utf-8")
            self.socket.sendall(struct.pack(">I", len(img_path_bytes)))
            self.socket.sendall(img_path_bytes)

            # Send text_query
            text_bytes = text_query.encode("utf-8")
            self.socket.sendall(struct.pack(">I", len(text_bytes)))
            self.socket.sendall(text_bytes)

            # Receive text chunks
            while True:
                # Receive chunk length
                length_data = self.recv_all(4)
                chunk_length = struct.unpack(">I", length_data)[0]

                # If length is 0, end of stream
                if chunk_length == 0:
                    break

                # Receive chunk data
                chunk_data = self.recv_all(chunk_length)
                chunk_text = chunk_data.decode("utf-8")

                yield chunk_text

        finally:
            self.close()

    def close(self):
        """Close connection."""
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None


def test_vlm_client(image_path: str, text_query: str = "What do you see?"):
    """Test VLM client with given image and query."""
    client = VLMClient()
    print(f"🤖 VLM Request: {text_query}")
    print(f"📷 Image: {image_path}")
    print("📝 Response: ", end="", flush=True)

    try:
        for chunk in client.generate(image_path, text_query):
            print(chunk, end="", flush=True)
        print("\n✅ VLM generation completed")
        return True
    except Exception as e:
        print(f"\n❌ VLM error: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vlm_client.py <image_path> [text_query]")
        sys.exit(1)

    image_path = sys.argv[1]
    text_query = sys.argv[2] if len(sys.argv) > 2 else "What do you see?"

    test_vlm_client(image_path, text_query)
