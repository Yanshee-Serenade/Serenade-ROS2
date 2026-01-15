"""
Client module for connecting to model servers.

Provides clients for:
- VLM Server: Vision Language Model inference
- DA3 Server: Depth Anything 3 depth estimation
- SAM3 Server: SAM3 segmentation
- Comparison: Depth comparison analysis
- Pointcloud: Point cloud validation
"""

from .comparison_client import run_comparison_client
from .config import client_config
from .da3_client import DA3Client
from .pointcloud_client import run_pointcloud_client
from .sam3_client import SAM3Client
from .vlm_client import VLMClient

__all__ = [
    "client_config",
    "VLMClient",
    "DA3Client",
    "SAM3Client",
    "run_comparison_client",
    "run_pointcloud_client",
]
