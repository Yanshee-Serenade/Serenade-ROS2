"""
Server module for AI vision models.

Provides three independent TCP servers:
- VLM Server (port 21122): Vision Language Model inference
- DA3 Server (port 21123): Depth Anything 3 depth estimation
- SAM3 Server (port 21124): SAM3 segmentation
"""

from .config import config
from .da3_server import run_da3_server
from .sam3_server import run_sam3_server
from .vlm_server import run_vlm_server

__all__ = [
    "config",
    "run_vlm_server",
    "run_da3_server",
    "run_sam3_server",
]
