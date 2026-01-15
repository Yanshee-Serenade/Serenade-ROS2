"""
Chatbot module for voice assistant functionality.

This module provides a modular system for voice-based interaction,
including speech recognition, LLM integration, and text-to-speech.
"""

from .asr import VoiceASR
from .assistant import VoiceAssistant
from .llm_client import LLMClient
from .main import SEGMENT_TTS, main
from .tts import StreamTTS

__all__ = [
    "VoiceASR",
    "LLMClient",
    "StreamTTS",
    "VoiceAssistant",
    "SEGMENT_TTS",
    "main",
]
