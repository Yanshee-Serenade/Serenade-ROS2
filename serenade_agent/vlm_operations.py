#!/usr/bin/env python3
"""
VLM operations module for model loading and inference.
Handles model initialization, tokenization, and generation.
"""

from typing import Any, Optional
import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)


class VLMOperations:
    """Manages VLM model operations and inference."""

    def __init__(self, model_name: str, max_new_tokens: int = 256):
        """
        Initialize VLM operations.
        
        Args:
            model_name: Name of the model to load
            max_new_tokens: Maximum number of tokens to generate
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.processor: Optional[Any] = None
        self.model_vlm: Optional[Any] = None

    def load_model(self, logger=None):
        """
        Load VLM model and processor.
        
        Args:
            logger: Optional ROS2 logger for logging
        """
        if logger:
            logger.info(f"Loading VLM model from {self.model_name}...")
        
        try:
            if logger:
                logger.info("Loading model processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)

            if logger:
                logger.info("Loading model weights...")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            self.model_vlm = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="cuda",
            )
            self.model_vlm.eval()
            if logger:
                logger.info("✅ VLM model loaded successfully!")
        except Exception as e:
            error_msg = f"VLM model loading failed: {str(e)}"
            if logger:
                logger.error(error_msg)
            raise

    def prepare_inputs(self, messages: list, logger=None):
        """
        Prepare inputs for model inference.
        
        Args:
            messages: List of messages in VLM format
            logger: Optional ROS2 logger for logging
            
        Returns:
            Prepared inputs as tensors on the correct device
        """
        if self.processor is None or self.model_vlm is None:
            raise ValueError("VLM model not loaded. Call load_model() first.")

        processor = self.processor

        # Apply chat template
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

        return inputs

    def get_streamer(self):
        """
        Get a TextIteratorStreamer for streaming generation.
        
        Returns:
            TextIteratorStreamer instance
        """
        if self.processor is None:
            raise ValueError("VLM model not loaded. Call load_model() first.")

        tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )
        return TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

    def get_generation_kwargs(self, inputs: dict, streamer) -> dict:
        """
        Build generation kwargs for model.generate().
        
        Args:
            inputs: Prepared inputs from prepare_inputs()
            streamer: TextIteratorStreamer from get_streamer()
            
        Returns:
            Dictionary of generation kwargs
        """
        return {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": True,
            "num_beams": 1,
        }

    def get_model(self):
        """
        Get the loaded model instance.
        
        Returns:
            The VLM model
        """
        if self.model_vlm is None:
            raise ValueError("VLM model not loaded. Call load_model() first.")
        return self.model_vlm
