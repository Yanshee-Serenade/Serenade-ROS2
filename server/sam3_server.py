"""
SAM3 Server - TCP server for SAM3 segmentation inference.
Listens on port 21124.
Uses binary protocol for efficient tensor transfer.
"""

import datetime
import socket
import struct
from threading import Thread

import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

from .config import config


class SAM3Server:
    """TCP server for SAM3 inference with binary protocol."""

    def __init__(self, host: str = config.FLASK_HOST, port: int = config.SAM3_PORT):
        self.host = host
        self.port = port
        self.processor = None

    def load_model(self):
        """Load SAM3 model."""
        print(f"{datetime.datetime.now()} > Loading SAM3 model...")
        model = build_sam3_image_model(
            load_from_HF=False,
            checkpoint_path=config.MODEL_SAM3_PATH,
        )
        self.processor = Sam3Processor(model)
        print(f"{datetime.datetime.now()} > ✅ SAM3 model ready!")

    def recv_all(self, conn: socket.socket, length: int) -> bytes:
        """Receive exact number of bytes."""
        data = b""
        while len(data) < length:
            chunk = conn.recv(min(length - len(data), 4096))
            if not chunk:
                raise ConnectionError("Connection closed unexpectedly")
            data += chunk
        return data

    def send_tensor(self, conn: socket.socket, tensor: torch.Tensor):
        """Send torch tensor as numpy array with shape and dtype info."""
        # Convert to numpy
        if tensor.is_cuda:
            arr = tensor.cpu().numpy()
        else:
            arr = tensor.numpy()

        # Send number of dimensions
        ndim = len(arr.shape)
        conn.sendall(struct.pack(">I", ndim))

        # Send each dimension size
        for dim in arr.shape:
            conn.sendall(struct.pack(">I", dim))

        # Send dtype as string
        dtype_str = str(arr.dtype)
        dtype_bytes = dtype_str.encode("utf-8")
        conn.sendall(struct.pack(">I", len(dtype_bytes)))
        conn.sendall(dtype_bytes)

        # Send array data
        arr_bytes = arr.tobytes()
        conn.sendall(struct.pack(">I", len(arr_bytes)))
        conn.sendall(arr_bytes)

    def handle_client(self, conn: socket.socket, addr):
        """Handle client connection."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[{timestamp}] 🔗 SAM3 client connected: {addr}")

        try:
            # Read image_path length
            img_path_len_data = self.recv_all(conn, 4)
            img_path_len = struct.unpack(">I", img_path_len_data)[0]

            # Read image_path
            img_path_data = self.recv_all(conn, img_path_len)
            image_path = img_path_data.decode("utf-8")

            # Read prompt length
            prompt_len_data = self.recv_all(conn, 4)
            prompt_len = struct.unpack(">I", prompt_len_data)[0]

            # Read prompt
            prompt_data = self.recv_all(conn, prompt_len)
            prompt = prompt_data.decode("utf-8")

            print(
                f"[{timestamp}] 🎯 SAM3 inference for: {image_path}, prompt: {prompt}"
            )

            # Run inference
            image = Image.open(image_path)
            if self.processor is None:
                raise ValueError("SAM3 processor not loaded")
            inference_state = self.processor.set_image(image)
            inference_state = self.processor.set_text_prompt(
                state=inference_state, prompt=prompt
            )

            # Send success flag
            conn.sendall(struct.pack(">B", 1))

            # Send original dimensions
            conn.sendall(struct.pack(">I", inference_state["original_height"]))
            conn.sendall(struct.pack(">I", inference_state["original_width"]))

            # Send tensors
            self.send_tensor(conn, inference_state["masks_logits"])
            self.send_tensor(conn, inference_state["masks"])
            self.send_tensor(conn, inference_state["boxes"])
            self.send_tensor(conn, inference_state["scores"])

            print(f"[{timestamp}] ✅ SAM3 inference completed")
            print(
                f"  • Masks shape: {inference_state['masks'].shape}, Boxes: {inference_state['boxes'].shape}"
            )

        except Exception as e:
            error_msg = f"SAM3 error: {str(e)}"
            print(f"[{timestamp}] ❌ {error_msg}")
            # Send failure flag
            try:
                conn.sendall(struct.pack(">B", 0))
                error_bytes = error_msg.encode("utf-8")
                conn.sendall(struct.pack(">I", len(error_bytes)))
                conn.sendall(error_bytes)
            except Exception:
                pass

        finally:
            conn.close()
            print(f"[{timestamp}] 🔌 SAM3 client disconnected")

    def run(self):
        """Run SAM3 TCP server."""
        self.load_model()

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)

        print(f"\n🚀 SAM3 TCP Server listening on {self.host}:{self.port}")

        try:
            while True:
                conn, addr = server_socket.accept()
                # Handle each client in a new thread
                client_thread = Thread(target=self.handle_client, args=(conn, addr))
                client_thread.daemon = True
                client_thread.start()
        except KeyboardInterrupt:
            print("\n🛑 SAM3 Server shutting down...")
        finally:
            server_socket.close()


def run_sam3_server(host: str = config.FLASK_HOST, port: int = config.SAM3_PORT):
    """Run SAM3 TCP server."""
    server = SAM3Server(host, port)
    server.run()


if __name__ == "__main__":
    run_sam3_server()
