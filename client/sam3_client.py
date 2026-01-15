"""
SAM3 Client - Client for SAM3 segmentation TCP server.
Connects to SAM3 server on port 21124 using binary protocol.
"""

import socket
import struct
from typing import NamedTuple, Optional

import numpy as np
import torch

from .config import client_config


class SAM3InferenceState(NamedTuple):
    """SAM3 inference state result."""

    original_height: int
    original_width: int
    masks_logits: torch.Tensor
    masks: torch.Tensor
    boxes: torch.Tensor
    scores: torch.Tensor


class SAM3Client:
    """Client for SAM3 TCP server with binary protocol."""

    def __init__(
        self,
        host: str = client_config.SAM3_HOST,
        port: int = client_config.SAM3_PORT,
        timeout: float = client_config.RECV_TIMEOUT,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket: Optional[socket.socket] = None

    def connect(self):
        """Connect to SAM3 server."""
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

    def recv_tensor(self) -> torch.Tensor:
        """Receive torch tensor (sent as numpy array)."""
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

        # Reconstruct numpy array
        arr = np.frombuffer(arr_data, dtype=np.dtype(dtype_str))
        arr = arr.reshape(shape)

        # Convert to torch tensor
        tensor = torch.from_numpy(arr.copy())

        return tensor

    def inference(self, image_path: str, prompt: str) -> Optional[SAM3InferenceState]:
        """
        Run SAM3 segmentation inference.

        Args:
            image_path: Path to image file
            prompt: Text prompt for segmentation

        Returns:
            SAM3InferenceState with masks, boxes, scores, etc.
        """
        try:
            self.connect()

            # Send image_path
            if self.socket is None:
                raise ConnectionError("Socket not connected")
            img_path_bytes = image_path.encode("utf-8")
            self.socket.sendall(struct.pack(">I", len(img_path_bytes)))
            self.socket.sendall(img_path_bytes)

            # Send prompt
            prompt_bytes = prompt.encode("utf-8")
            self.socket.sendall(struct.pack(">I", len(prompt_bytes)))
            self.socket.sendall(prompt_bytes)

            # Receive success flag
            success_data = self.recv_all(1)
            success = struct.unpack(">B", success_data)[0] == 1

            if not success:
                # Receive error message
                error_len_data = self.recv_all(4)
                error_len = struct.unpack(">I", error_len_data)[0]
                error_data = self.recv_all(error_len)
                error_msg = error_data.decode("utf-8")
                raise Exception(f"SAM3 server error: {error_msg}")

            # Receive original dimensions
            height_data = self.recv_all(4)
            original_height = struct.unpack(">I", height_data)[0]

            width_data = self.recv_all(4)
            original_width = struct.unpack(">I", width_data)[0]

            # Receive tensors
            masks_logits = self.recv_tensor()
            masks = self.recv_tensor()
            boxes = self.recv_tensor()
            scores = self.recv_tensor()

            return SAM3InferenceState(
                original_height=original_height,
                original_width=original_width,
                masks_logits=masks_logits,
                masks=masks,
                boxes=boxes,
                scores=scores,
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


def test_sam3_client(image_path: str, prompt: str = "object"):
    """Test SAM3 client with given image and prompt."""
    client = SAM3Client()
    print(f"🎯 SAM3 Request for: {image_path}")
    print(f"💬 Prompt: {prompt}")

    try:
        state = client.inference(image_path, prompt)
        if state:
            print("✅ SAM3 inference completed")
            print(f"  • Original size: {state.original_height}x{state.original_width}")
            print(f"  • Masks logits shape: {state.masks_logits.shape}")
            print(f"  • Masks shape: {state.masks.shape}")
            print(f"  • Boxes shape: {state.boxes.shape}")
            print(f"  • Scores shape: {state.scores.shape}")
            return True
        else:
            print("❌ SAM3 inference failed")
            return False
    except Exception as e:
        print(f"❌ SAM3 error: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python sam3_client.py <image_path> [prompt]")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "object"

    test_sam3_client(image_path, prompt)
