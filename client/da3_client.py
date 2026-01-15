"""
DA3 Client - Client for Depth Anything 3 TCP server.
Connects to DA3 server on port 21123 using binary protocol.
"""

import socket
import struct
from typing import NamedTuple, Optional

import numpy as np

from .config import client_config


class DA3Prediction(NamedTuple):
    """DA3 prediction result."""

    depth: np.ndarray
    conf: np.ndarray
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    processed_images: np.ndarray


class DA3Client:
    """Client for DA3 TCP server with binary protocol."""

    def __init__(
        self,
        host: str = client_config.DA3_HOST,
        port: int = client_config.DA3_PORT,
        timeout: float = client_config.RECV_TIMEOUT,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket: Optional[socket.socket] = None

    def connect(self):
        """Connect to DA3 server."""
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

    def recv_array(self) -> np.ndarray:
        """Receive numpy array with shape and dtype info."""
        # Receive number of dimensions
        ndim_data = self.recv_all(4)
        ndim = struct.unpack(">I", ndim_data)[0]

        # Receive shape
        shape = []
        for _ in range(ndim):
            dim_data = self.recv_all(4)
            dim = struct.unpack(">I", dim_data)[0]
            shape.append(dim)

        # Receive dtype
        dtype_len_data = self.recv_all(4)
        dtype_len = struct.unpack(">I", dtype_len_data)[0]
        dtype_data = self.recv_all(dtype_len)
        dtype_str = dtype_data.decode("utf-8")

        # Receive array data
        arr_len_data = self.recv_all(4)
        arr_len = struct.unpack(">I", arr_len_data)[0]
        arr_data = self.recv_all(arr_len)

        # Reconstruct array
        arr = np.frombuffer(arr_data, dtype=np.dtype(dtype_str))
        arr = arr.reshape(shape)

        return arr

    def inference(self, image_path: str) -> Optional[DA3Prediction]:
        """
        Run DA3 depth inference.

        Args:
            image_path: Path to image file

        Returns:
            DA3Prediction with depth, conf, extrinsics, intrinsics, processed_images
        """
        try:
            self.connect()

            # Send image_path
            if self.socket is None:
                raise ConnectionError("Socket not connected")
            img_path_bytes = image_path.encode("utf-8")
            self.socket.sendall(struct.pack(">I", len(img_path_bytes)))
            self.socket.sendall(img_path_bytes)

            # Receive success flag
            success_data = self.recv_all(1)
            success = struct.unpack(">B", success_data)[0] == 1

            if not success:
                # Receive error message
                error_len_data = self.recv_all(4)
                error_len = struct.unpack(">I", error_len_data)[0]
                error_data = self.recv_all(error_len)
                error_msg = error_data.decode("utf-8")
                raise Exception(f"DA3 server error: {error_msg}")

            # Receive arrays
            depth = self.recv_array()
            conf = self.recv_array()
            extrinsics = self.recv_array()
            intrinsics = self.recv_array()
            processed_images = self.recv_array()

            return DA3Prediction(
                depth=depth,
                conf=conf,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                processed_images=processed_images,
            )

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


def test_da3_client(image_path: str):
    """Test DA3 client with given image."""
    client = DA3Client()
    print(f"📊 DA3 Request for: {image_path}")

    try:
        prediction = client.inference(image_path)
        if prediction:
            print("✅ DA3 inference completed")
            print(f"  • Depth shape: {prediction.depth.shape}")
            print(f"  • Conf shape: {prediction.conf.shape}")
            print(f"  • Extrinsics shape: {prediction.extrinsics.shape}")
            print(f"  • Intrinsics shape: {prediction.intrinsics.shape}")
            print(f"  • Processed images shape: {prediction.processed_images.shape}")
            return True
        else:
            print("❌ DA3 inference failed")
            return False
    except Exception as e:
        print(f"❌ DA3 error: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python da3_client.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    test_da3_client(image_path)
