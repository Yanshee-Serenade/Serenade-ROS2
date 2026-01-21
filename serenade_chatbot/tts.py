"""
Text-to-speech module for voice assistant.
This module provides the StreamTTS class for handling text-to-speech
with streaming and interrupt support.
"""
import asyncio
import time
import serenade_chatbot.YanAPI as YanAPI

class StreamTTS:
    """Stream TTS Wrapper with Buffering and Chinese Sentence Segmentation"""
    
    # Chinese punctuation marks that can end a sentence
    CHINESE_SENTENCE_ENDERS = {"。", "！", "？", "；"}
    CHINESE_CONTINUATION_MARKS = {"，", "、"}
    
    def __init__(self):
        # Buffer to store text chunks while TTS is playing
        self.text_buffer = []
        self.is_running = False
        self.partial_sentence = ""  # For accumulating incomplete sentences
    
    def _split_by_punctuation(self, text: str):
        """Split text by Chinese punctuation marks, preserving the punctuation."""
        result = []
        current_segment = ""
        
        for char in text:
            current_segment += char
            # Check if character is a sentence-ending punctuation
            if char in self.CHINESE_SENTENCE_ENDERS:
                result.append(current_segment)
                current_segment = ""
            # Check for continuation marks (comma-like punctuation)
            elif char in self.CHINESE_CONTINUATION_MARKS:
                result.append(current_segment)
                current_segment = ""
        
        # Add any remaining text as a partial sentence
        if current_segment:
            result.append(current_segment)
            
        return result
    
    def _find_last_sentence_boundary(self, text: str):
        """Find the last complete sentence boundary in the text."""
        # Look for the last occurrence of any sentence-ending punctuation
        for i in range(len(text) - 1, -1, -1):
            if text[i] in self.CHINESE_SENTENCE_ENDERS:
                return i + 1  # Include the punctuation
        
        # If no sentence ender found, look for continuation marks
        for i in range(len(text) - 1, -1, -1):
            if text[i] in self.CHINESE_CONTINUATION_MARKS:
                return i + 1  # Include the punctuation
        
        return 0  # No boundary found
    
    async def add_text(self, text: str, interrupt: bool = False):
        """
        Add text to the TTS buffer with Chinese sentence segmentation.
        If a chunk starts with a Chinese sentence-ending punctuation,
        we split it and send the previous buffer with that punctuation.
        """
        if interrupt:
            # Clear pending text so we don't say old stuff after interruption
            self.text_buffer.clear()
            self.partial_sentence = ""
            
            # Stop the physical TTS playback immediately
            try:
                YanAPI.stop_voice_tts()
            except Exception as e:
                print(f"Error stopping TTS: {e}")
        
        if not text:
            return
        
        # Check if the text starts with a sentence-ending punctuation
        if text and text[0] in self.CHINESE_SENTENCE_ENDERS:
            # This means the previous text should have ended with this punctuation
            if self.partial_sentence:
                # Add the punctuation to complete the previous sentence
                complete_sentence = self.partial_sentence + text[0]
                print(f"Completing sentence with punctuation: {complete_sentence}")
                self.text_buffer.append(complete_sentence)
                self.partial_sentence = ""
                text = text[1:]  # Remove the punctuation we just used
            else:
                # Just add the punctuation character
                self.text_buffer.append(text[0])
                text = text[1:]
        
        # Process the remaining text
        if text:
            print(f"Adding text to buffer: {text}")
            
            # Split the text by punctuation boundaries
            segments = self._split_by_punctuation(text)
            
            # Add complete sentences to buffer, keep partial ones for accumulation
            for i, segment in enumerate(segments):
                # If this segment ends with sentence punctuation, it's complete
                if segment and segment[-1] in (self.CHINESE_SENTENCE_ENDERS | self.CHINESE_CONTINUATION_MARKS):
                    # If we have a partial sentence from before, combine it
                    if self.partial_sentence:
                        complete_segment = self.partial_sentence + segment
                        self.text_buffer.append(complete_segment)
                        self.partial_sentence = ""
                    else:
                        self.text_buffer.append(segment)
                else:
                    # This is a partial sentence without ending punctuation
                    if i == len(segments) - 1:  # Only last segment can be partial
                        # Check if it starts with a new sentence marker like "然后"
                        # In this case, the previous text should have had punctuation
                        if self.partial_sentence:
                            # Combine with existing partial sentence
                            self.partial_sentence += segment
                        else:
                            self.partial_sentence = segment
                    else:
                        # Should not happen since _split_by_punctuation only returns
                        # incomplete sentences as the last segment
                        self.text_buffer.append(segment)
        
        # Start the consumer loop if it's not already running
        if not self.is_running and (self.text_buffer or self.partial_sentence):
            asyncio.create_task(self._process_buffer())
    
    async def _process_buffer(self):
        """Continuously processes the text buffer."""
        self.is_running = True
        
        while self.text_buffer or self.partial_sentence:
            # First, check if we have any complete sentences in buffer
            if self.text_buffer:
                # Join all complete sentences
                current_text_chunk = "".join(self.text_buffer)
                self.text_buffer.clear()
                
                print(f"Speaking complete sentence(s): {current_text_chunk}")
                
                if current_text_chunk:
                    try:
                        # Send to TTS
                        timestamp = int(time.time())
                        result = YanAPI.start_voice_tts(
                            tts=current_text_chunk,
                            interrupt=False,
                            timestamp=timestamp
                        )
                        
                        if result["code"] == 0:
                            await self._wait_for_completion(timestamp)
                    except Exception as e:
                        print(f"TTS Error: {e}")
                        await asyncio.sleep(1)
            
            # If no complete sentences in buffer but we have a partial sentence,
            # check if we should wait for more text or send it anyway
            elif self.partial_sentence:
                # Wait a bit to see if more text comes in
                await asyncio.sleep(1)
                
                # If we've been waiting and no new text came in,
                # we might want to send the partial sentence anyway
                # This prevents hanging on incomplete sentences
                if self.partial_sentence and not self.text_buffer:
                    # Force send partial sentence (add a period to make it complete)
                    forced_sentence = self.partial_sentence + "。"
                    print(f"Forcing partial sentence: {forced_sentence}")
                    
                    try:
                        timestamp = int(time.time())
                        result = YanAPI.start_voice_tts(
                            tts=forced_sentence,
                            interrupt=False,
                            timestamp=timestamp
                        )
                        
                        if result["code"] == 0:
                            await self._wait_for_completion(timestamp)
                    except Exception as e:
                        print(f"TTS Error: {e}")
                    
                    self.partial_sentence = ""
        
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
    
    def flush(self):
        """Force sending any remaining partial sentence."""
        if self.partial_sentence:
            # Add a period to complete the sentence
            complete_sentence = self.partial_sentence + "。"
            self.text_buffer.append(complete_sentence)
            self.partial_sentence = ""
            
            if not self.is_running:
                asyncio.create_task(self._process_buffer())