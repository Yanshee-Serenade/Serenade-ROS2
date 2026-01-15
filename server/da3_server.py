"""
DA3 Server - TCP server for Depth Anything 3 inference.
Listens on port 21123.
Uses binary protocol for efficient numpy array transfer.
"""

import datetime
import socket
import struct
from threading import Thread

import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

from .config import config


class DA3Server:
    """TCP server for DA3 inference with binary protocol."""

    def __init__(self, host: str = config.FLASK_HOST, port: int = config.DA3_PORT):
        self.host = host
        self.port = port
        self.model_da3 = None

    def load_model(self):
        """Load DA3 model."""
        print(f"{datetime.datetime.now()} > Loading DA3 model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_da3 = DepthAnything3.from_pretrained(config.MODEL_DA3_DEFAULT)
        self.model_da3 = self.model_da3.to(device=device, dtype=torch.float32)
        self.model_da3.eval()
        if torch.cuda.is_available():
            self.model_da3 = torch.compile(self.model_da3, mode="reduce-overhead")
        print(f"{datetime.datetime.now()} > ✅ DA3 model ready!")

    def recv_all(self, conn: socket.socket, length: int) -> bytes:
        """Receive exact number of bytes."""
        data = b""
        while len(data) < length:
            chunk = conn.recv(min(length - len(data), 4096))
            if not chunk:
                raise ConnectionError("Connection closed unexpectedly")
            data += chunk
        return data

    def send_array(self, conn: socket.socket, arr: np.ndarray):
        """Send numpy array with shape and dtype info."""
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
        print(f"[{timestamp}] 🔗 DA3 client connected: {addr}")

        try:
            # Read image_path length
            img_path_len_data = self.recv_all(conn, 4)
            img_path_len = struct.unpack(">I", img_path_len_data)[0]

            # Read image_path
            img_path_data = self.recv_all(conn, img_path_len)
            image_path = img_path_data.decode("utf-8")

            print(f"[{timestamp}] 📊 DA3 inference for: {image_path}")

            # Run inference
            if self.model_da3 is None:
                raise ValueError("DA3 model not loaded")
            prediction = self.model_da3.inference(  # type: ignore[attr-defined]
                image=[image_path],
                process_res=504,
                process_res_method="upper_bound_resize",
                export_dir=None,
                export_format="glb",
            )

            # Send success flag
            conn.sendall(struct.pack(">B", 1))

            # Send each array
            if prediction.depth is not None:
                self.send_array(conn, prediction.depth)
            if prediction.conf is not None:
                self.send_array(conn, prediction.conf)
            if prediction.extrinsics is not None:
                self.send_array(conn, prediction.extrinsics)
            if prediction.intrinsics is not None:
                self.send_array(conn, prediction.intrinsics)
            if prediction.processed_images is not None:
                self.send_array(conn, prediction.processed_images)

            print(f"[{timestamp}] ✅ DA3 inference completed")
            if prediction.depth is not None:
                print(f"  • Depth shape: {prediction.depth.shape}")
            if prediction.conf is not None:
                print(f"  • Conf shape: {prediction.conf.shape}")

        except Exception as e:
            error_msg = f"DA3 error: {str(e)}"
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
            print(f"[{timestamp}] 🔌 DA3 client disconnected")

    def run(self):
        """Run DA3 TCP server."""
        self.load_model()

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)

        print(f"\n🚀 DA3 TCP Server listening on {self.host}:{self.port}")

        try:
            while True:
                conn, addr = server_socket.accept()
                # Handle each client in a new thread
                client_thread = Thread(target=self.handle_client, args=(conn, addr))
                client_thread.daemon = True
                client_thread.start()
        except KeyboardInterrupt:
            print("\n🛑 DA3 Server shutting down...")
        finally:
            server_socket.close()


def run_da3_server(host: str = config.FLASK_HOST, port: int = config.DA3_PORT):
    """Run DA3 TCP server."""
    server = DA3Server(host, port)
    server.run()


if __name__ == "__main__":
    run_da3_server()
