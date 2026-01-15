"""
VLM Client - Client for Vision Language Model TCP server.
Connects to VLM server on port 21122 using binary protocol.
Falls back to Doubao API (using Base64) if local server is unavailable.
"""

import base64
import os
import socket
import struct
from typing import Generator, Optional

from dotenv import load_dotenv

# --- VolcEngine Imports (Assumed available per requirements) ---
from volcenginesdkarkruntime import Ark
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

    def _encode_file_to_base64(self, file_path: str) -> str:
        """Helper to read file and convert to base64 string."""
        with open(file_path, "rb") as read_file:
            return base64.b64encode(read_file.read()).decode("utf-8")

    def _get_image_mime_type(self, file_path: str) -> str:
        """Simple helper to guess mime type from extension."""
        if file_path.lower().endswith(".png"):
            return "image/png"
        elif file_path.lower().endswith((".jpg", ".jpeg")):
            return "image/jpeg"
        elif file_path.lower().endswith(".webp"):
            return "image/webp"
        return "image/jpeg"  # Default fallback

    def fallback_to_doubao(
        self, image_path: str, text_query: str
    ) -> Generator[str, None, None]:
        """
        Executes the Doubao API logic synchronously using Base64 and yields the result.
        """
        print("(⚠️ Local VLM server unavailable, falling back to Doubao API...)")

        api_key = os.getenv("ARK_API_KEY")
        if not api_key:
            yield "Error: ARK_API_KEY environment variable is not set."
            return

        try:
            # 1. Encode Image to Base64
            base64_image = self._encode_file_to_base64(image_path)
            mime_type = self._get_image_mime_type(image_path)

            # 2. Initialize Client
            client = Ark(base_url=client_config.DOUBAO_BASE_URL, api_key=api_key)

            # 3. Create Stream
            stream = client.responses.create(
                model=client_config.DOUBAO_MODEL_ENDPOINT,
                input=[
                    {  # pyright: ignore[reportArgumentType]
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:{mime_type};base64,{base64_image}",
                            },
                            {"type": "input_text", "text": text_query},
                        ],
                    }
                ],
                stream=True,
            )

            # 4. Process Stream
            for event in stream:
                if isinstance(event, ResponseReasoningSummaryTextDeltaEvent):
                    yield event.delta
                elif isinstance(event, ResponseTextDeltaEvent):
                    yield event.delta
                elif isinstance(event, ResponseOutputItemAddedEvent):
                    # Usually metadata, skipping for pure text generation stream
                    pass
                elif isinstance(event, ResponseTextDoneEvent):
                    # End of specific block
                    pass
                elif isinstance(event, ResponseCompletedEvent):
                    # Final usage stats
                    pass

        except FileNotFoundError:
            yield f"Error: Image file not found at {image_path}"
        except Exception as e:
            yield f"Doubao API Error: {str(e)}"

    def generate(self, image_path: str, text_query: str) -> Generator[str, None, None]:
        """
        Generate text from VLM.
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
                length_data = self.recv_all(4)
                chunk_length = struct.unpack(">I", length_data)[0]

                if chunk_length == 0:
                    break

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
