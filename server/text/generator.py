"""
Text generator module for Vision Language Model text generation.

This module provides functions for generating text responses from
Vision Language Models with streaming support.
"""

import json
from threading import Thread
from typing import Generator, Optional

import torch
from transformers import TextIteratorStreamer

from ..config import config
from ..models.model_loader import ModelManager


def generate_text_stream(
    text_query: str,
    image_path: str,
    timestamp: str,
    model_manager: Optional[ModelManager] = None,
) -> Generator[str, None, None]:
    """
    Generate streaming text response from VLM.

    Args:
        text_query: Text query/instruction
        image_path: Image file path
        timestamp: Timestamp for logging
        model_manager: ModelManager instance with VLM loaded. If None, creates new one.

    Yields:
        JSON-encoded SSE events with generated text
    """
    # Get model and processor
    if model_manager is None:
        yield f"data: {json.dumps({'text': '❌ Model manager not provided'})}\n\n"
        return

    if not model_manager.is_vlm_loaded():
        yield f"data: {json.dumps({'text': '❌ VLM model not loaded'})}\n\n"
        return

    processor = model_manager.get_processor()
    model_vlm = model_manager.get_vlm()

    try:
        print(f"[{timestamp}] 🤖 Starting text generation, query: '{text_query}'")

        # 1. Build conversation messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": image_path},
                    {"type": "text", "text": text_query},
                ],
            },
        ]

        # 2. Apply chat template and encode
        # Note: Different models have different ways to process inputs
        # For Qwen models, we need to use the tokenizer directly
        if hasattr(processor, "apply_chat_template"):
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            # Fallback: use tokenizer directly
            tokenizer = (
                processor.tokenizer if hasattr(processor, "tokenizer") else processor
            )
            text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = tokenizer(text, return_tensors="pt")

        # 3. Move inputs to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = inputs.to(
            device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        # 4. Initialize streaming generator
        tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # 5. Build generation parameters
        generation_kwargs = {
            "inputs": inputs,
            "streamer": streamer,
            "max_new_tokens": config.MAX_NEW_TOKENS,
            "do_sample": True,
            "num_beams": 1,
        }

        # 6. Start generation in separate thread
        def generate_wrapper():
            model_vlm.generate(**generation_kwargs)

        thread = Thread(target=generate_wrapper)
        thread.start()

        # 7. Stream generated results
        for new_text in streamer:
            if new_text:
                yield f"data: {json.dumps({'text': new_text})}\n\n"

        print(f"[{timestamp}] ✅ Text generation completed")

    except Exception as e:
        error_msg = f"Text generation failed: {str(e)}"
        print(f"[{timestamp}] ❌ {error_msg}")
        yield f"data: {json.dumps({'text': f'❌ {error_msg}'})}\n\n"


def generate_text_batch(
    text_query: str,
    image_path: str,
    model_manager: ModelManager,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
) -> str:
    """
    Generate text response in batch mode (non-streaming).

    Args:
        text_query: Text query/instruction
        image_path: Image file path
        model_manager: ModelManager instance with VLM loaded
        max_new_tokens: Maximum number of new tokens to generate

    Returns:
        Generated text response

    Raises:
        ValueError: If VLM model not loaded
        Exception: If text generation fails
    """
    if not model_manager.is_vlm_loaded():
        raise ValueError("VLM model not loaded")

    processor = model_manager.get_processor()
    model_vlm = model_manager.get_vlm()

    try:
        # Build conversation messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": image_path},
                    {"type": "text", "text": text_query},
                ],
            },
        ]

        # Apply chat template and encode
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
                processor.tokenizer if hasattr(processor, "tokenizer") else processor
            )
            text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = tokenizer(text, return_tensors="pt")

        # Move inputs to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = inputs.to(
            device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        # Generate text
        with torch.no_grad():
            outputs = model_vlm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                num_beams=1,
            )

        # Decode generated text
        tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        return generated_text

    except Exception as e:
        raise Exception(f"Batch text generation failed: {str(e)}")


def validate_text_query(text_query: str) -> tuple[bool, str]:
    """
    Validate text query.

    Args:
        text_query: Text query to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text_query or not text_query.strip():
        return False, "Text query cannot be empty"

    if len(text_query.strip()) > 1000:
        return False, "Text query too long (max 1000 characters)"

    return True, ""
