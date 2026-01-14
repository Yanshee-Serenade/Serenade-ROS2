"""
Text generation module for VLM text streaming.
Handles vision-language model text generation with streaming support.
"""

import json
from threading import Thread
from typing import Dict, Generator, Optional, Tuple

import torch
from transformers import TextIteratorStreamer

# Import models and configuration
from .config import MAX_NEW_TOKENS
from .models import get_vlm_model, get_vlm_processor


def prepare_vlm_input(
    image_path: str,
    text_query: str,
    processor,
    model,
    add_generation_prompt: bool = True,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Prepare input for VLM model.

    Args:
        image_path: Path to input image
        text_query: Text query/prompt
        processor: VLM processor instance
        model: VLM model instance
        add_generation_prompt: Whether to add generation prompt

    Returns:
        Prepared model inputs or None on failure
    """
    try:
        # Create message structure for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_query},
                ],
            }
        ]

        # Apply chat template and tokenize
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move to appropriate device and dtype
        device = (
            next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
        )
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        inputs = {k: v.to(device, dtype=dtype) for k, v in inputs.items()}

        return inputs

    except Exception as e:
        print(f"❌ Failed to prepare VLM input: {str(e)}")
        return None


def create_streamer(processor) -> TextIteratorStreamer:
    """
    Create text streamer for streaming generation.

    Args:
        processor: VLM processor with tokenizer

    Returns:
        TextIteratorStreamer instance
    """
    return TextIteratorStreamer(
        getattr(processor, "tokenizer", processor),
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=None,
    )


def generate_text_stream(
    text_query: str,
    image_path: str,
    timestamp: str,
) -> Generator[str, None, None]:
    """
    Generate text stream from VLM model.

    Args:
        text_query: Text query/prompt
        image_path: Path to input image
        timestamp: Timestamp for logging

    Yields:
        SSE formatted text chunks
    """
    # Get model and processor
    model = get_vlm_model()
    processor = get_vlm_processor()

    if model is None or processor is None:
        error_msg = "VLM model or processor not initialized"
        yield f"data: {json.dumps({'text': f'❌ {error_msg}'})}\n\n"
        return

    try:
        print(f"[{timestamp}] 🔍 Starting text generation...")
        print(f"[{timestamp}] 📝 Query: {text_query}")

        # Prepare model inputs
        inputs = prepare_vlm_input(image_path, text_query, processor, model)
        if inputs is None:
            error_msg = "Failed to prepare model inputs"
            yield f"data: {json.dumps({'text': f'❌ {error_msg}'})}\n\n"
            return

        # Create streamer
        streamer = create_streamer(processor)

        # Prepare generation parameters
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": True,
            "num_beams": 1,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        # Start generation in separate thread
        thread = Thread(target=lambda: model.generate(**generation_kwargs))
        thread.start()

        # Stream results
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
    max_new_tokens: Optional[int] = None,
) -> Tuple[Optional[str], Optional[Dict[str, object]]]:
    """
    Generate text in batch mode (non-streaming).

    Args:
        text_query: Text query/prompt
        image_path: Path to input image
        max_new_tokens: Maximum new tokens to generate

    Returns:
        Tuple of (generated_text, generation_info) or (None, None) on failure
    """
    # Get model and processor
    model = get_vlm_model()
    processor = get_vlm_processor()

    if model is None or processor is None:
        print("❌ VLM model or processor not initialized")
        return None, None

    try:
        # Prepare model inputs
        inputs = prepare_vlm_input(image_path, text_query, processor, model)
        if inputs is None:
            return None, None

        # Generate text
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens or MAX_NEW_TOKENS,
            "do_sample": True,
            "num_beams": 1,
            "temperature": 0.7,
            "top_p": 0.9,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)

        # Decode generated text
        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1] :]
        tokenizer = getattr(processor, "tokenizer", processor)
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Prepare generation info
        generation_info = {
            "generated_tokens": len(generated_ids),
            "max_new_tokens": generation_kwargs["max_new_tokens"],
            "has_scores": outputs.scores is not None,
        }

        return generated_text, generation_info

    except Exception as e:
        print(f"❌ Batch text generation failed: {str(e)}")
        return None, None


def validate_text_query(text_query: str) -> Tuple[bool, Optional[str]]:
    """
    Validate text query.

    Args:
        text_query: Text query to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text_query or not isinstance(text_query, str):
        return False, "Text query must be a non-empty string"

    if len(text_query.strip()) == 0:
        return False, "Text query cannot be empty or whitespace only"

    if len(text_query) > 1000:
        return False, "Text query too long (max 1000 characters)"

    return True, None


def format_error_response(error_message: str) -> str:
    """
    Format error response for streaming.

    Args:
        error_message: Error message to format

    Returns:
        SSE formatted error response
    """
    return f"data: {json.dumps({'text': f'❌ {error_message}'})}\n\n"


def format_success_response(text: str) -> str:
    """
    Format success response for streaming.

    Args:
        text: Text to format

    Returns:
        SSE formatted success response
    """
    return f"data: {json.dumps({'text': text})}\n\n"
