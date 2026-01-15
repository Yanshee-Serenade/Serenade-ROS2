"""
VLM Server - TCP server for Vision Language Model inference.
Listens on port 21122.
Uses binary protocol for efficient data transfer.
"""

import datetime
import socket
import struct
import time
from threading import Thread
from typing import Any, Optional

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from .config import config


class VLMServer:
    """TCP server for VLM inference with binary protocol."""

    def __init__(self, host: str = config.FLASK_HOST, port: int = config.VLM_PORT):
        self.host = host
        self.port = port
        self.processor: Optional[Any] = None
        self.model_vlm: Optional[Any] = None

    def load_model(self, model_path: str = config.MODEL_VLM_DEFAULT):
        """Load VLM model."""
        print(f"{datetime.datetime.now()} > Loading VLM model...")
        try:
            # Load processor
            print(
                f"{time.time()} > Loading model processor: {model_path}...", flush=True
            )
            self.processor = AutoProcessor.from_pretrained(model_path)

            # Load model
            print(f"{time.time()} > Loading model weights...", flush=True)
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            self.model_vlm = AutoModelForImageTextToText.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
            )
            self.model_vlm.eval()
            print(f"{datetime.datetime.now()} > ✅ VLM model ready!")
        except Exception as e:
            raise Exception(f"VLM model loading failed: {str(e)}")

    def recv_all(self, conn: socket.socket, length: int) -> bytes:
        """Receive exact number of bytes."""
        data = b""
        while len(data) < length:
            chunk = conn.recv(min(length - len(data), 4096))
            if not chunk:
                raise ConnectionError("Connection closed unexpectedly")
            data += chunk
        return data

    def handle_client(self, conn: socket.socket, addr):
        """Handle client connection."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[{timestamp}] 🔗 VLM client connected: {addr}")

        try:
            # Read image_path length
            img_path_len_data = self.recv_all(conn, 4)
            img_path_len = struct.unpack(">I", img_path_len_data)[0]

            # Read image_path
            img_path_data = self.recv_all(conn, img_path_len)
            image_path = img_path_data.decode("utf-8")

            # Read text query length
            text_len_data = self.recv_all(conn, 4)
            text_len = struct.unpack(">I", text_len_data)[0]

            # Read text query
            text_data = self.recv_all(conn, text_len)
            text_query = text_data.decode("utf-8")

            print(
                f"[{timestamp}] 🤖 VLM request: image={image_path}, text={text_query}"
            )

            # Build messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "path": image_path},
                        {"type": "text", "text": text_query},
                    ],
                }
            ]

            # Get model components
            if self.processor is None or self.model_vlm is None:
                raise ValueError("VLM model not loaded")
            processor = self.processor
            model_vlm = self.model_vlm

            # Prepare inputs
            if hasattr(processor, "apply_chat_template"):
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            else:
                tokenizer = (
                    processor.tokenizer
                    if hasattr(processor, "tokenizer")
                    else processor
                )
                text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                inputs = tokenizer(text, return_tensors="pt")

            # Move to device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            inputs = inputs.to(device=device, dtype=dtype)

            # Setup streamer
            tokenizer = (
                processor.tokenizer if hasattr(processor, "tokenizer") else processor
            )
            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            # Generation parameters
            generation_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": config.MAX_NEW_TOKENS,
                "do_sample": True,
                "num_beams": 1,
            }

            # Start generation thread
            generation_thread = Thread(
                target=lambda: model_vlm.generate(**generation_kwargs)
            )
            generation_thread.start()

            # Stream response as text chunks
            for text in streamer:
                if text:
                    text_bytes = text.encode("utf-8")
                    # Send chunk length + chunk data
                    conn.sendall(struct.pack(">I", len(text_bytes)))
                    conn.sendall(text_bytes)

            # Send end marker (length = 0)
            conn.sendall(struct.pack(">I", 0))

            print(f"[{timestamp}] ✅ VLM generation completed")

        except Exception as e:
            error_msg = f"VLM error: {str(e)}"
            print(f"[{timestamp}] ❌ {error_msg}")
            # Send error as single chunk then end marker
            try:
                error_bytes = error_msg.encode("utf-8")
                conn.sendall(struct.pack(">I", len(error_bytes)))
                conn.sendall(error_bytes)
                conn.sendall(struct.pack(">I", 0))
            except Exception:
                pass

        finally:
            try:
                conn.close()
            except Exception:
                pass
            print(f"[{timestamp}] 🔌 VLM client disconnected")

    def run(self):
        """Run VLM TCP server."""
        self.load_model()

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)

        print(f"\n🚀 VLM TCP Server listening on {self.host}:{self.port}")

        try:
            while True:
                conn, addr = server_socket.accept()
                # Handle each client in a new thread
                client_thread = Thread(target=self.handle_client, args=(conn, addr))
                client_thread.daemon = True
                client_thread.start()
        except KeyboardInterrupt:
            print("\n🛑 VLM Server shutting down...")
        finally:
            server_socket.close()


def run_vlm_server(host: str = config.FLASK_HOST, port: int = config.VLM_PORT):
    """Run VLM TCP server."""
    server = VLMServer(host, port)
    server.run()


if __name__ == "__main__":
    run_vlm_server()
