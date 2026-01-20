"""
Text-to-speech module for voice assistant.
This module provides the StreamTTS class for handling text-to-speech
with streaming and interrupt support.
"""
import asyncio
import time
import serenade_chatbot.YanAPI as YanAPI

class StreamTTS:
    """Stream TTS Wrapper with Buffering"""

    def __init__(self):
        # Buffer to store text chunks while TTS is playing
        self.text_buffer = []
        self.is_running = False

    async def add_text(self, text: str, interrupt: bool = False):
        """
        Add text to the TTS buffer.
        
        If interrupt is True:
        1. Clears the current buffer.
        2. Stops current playback immediately.
        3. Sets the new text as the next thing to play.
        """
        if interrupt:
            # Clear pending text so we don't say old stuff after interruption
            self.text_buffer.clear()
            # Stop the physical TTS playback immediately
            try:
                YanAPI.stop_voice_tts()
            except Exception as e:
                print(f"Error stopping TTS: {e}")
        
        # Add the new text to the buffer
        if text:
            print(f"Adding buffer {text}")
            self.text_buffer.append(text)

        # Start the consumer loop if it's not already running
        if not self.is_running:
            asyncio.create_task(self._process_buffer())

    async def _process_buffer(self):
        """
        Continuously processes the text buffer. 
        It accumulates text while playing, then sends the accumulated batch.
        """
        self.is_running = True

        while self.text_buffer:
            # 1. Accumulate: Extract all currently available text from buffer
            # We join them to create a larger chunk, reducing TTS calls and stutter
            current_text_chunk = "".join(self.text_buffer)
            self.text_buffer.clear() # Clear buffer for new incoming chunks
            print(f"Speaking {current_text_chunk}")

            if not current_text_chunk:
                continue

            try:
                # 2. Play: Send the accumulated chunk to TTS
                timestamp = int(time.time())
                
                # We pass interrupt=False here because we handled the 
                # interrupt logic explicitly in add_text
                result = YanAPI.start_voice_tts(
                    tts=current_text_chunk, 
                    interrupt=False, 
                    timestamp=timestamp
                )

                if result["code"] == 0:
                    # 3. Wait: While we wait here, add_text can keep filling self.text_buffer
                    await self._wait_for_completion(timestamp)

            except Exception as e:
                print(f"TTS Error: {e}")
                # Optional: slight delay on error to prevent tight loops
                await asyncio.sleep(0.1)

        self.is_running = False

    async def _wait_for_completion(self, timestamp: int):
        """Wait for TTS playback to finish"""
        while True:
            try:
                status = YanAPI.get_voice_tts_state(timestamp)
                # If status is idle, playback finished (or was interrupted externally)
                if status.get("status") == "idle":
                    break
                
                await asyncio.sleep(0.5)
            except Exception:
                break
        await asyncio.sleep(0.5)
