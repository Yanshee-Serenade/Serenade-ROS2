#!/usr/bin/env python3
"""
Message builder for VLM inference with conversation history management.
Maintains stateful conversation history and handles image integration.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional
import cv2


SYSTEM_PROMPT = """
你是一个机器人，图片是你现在看到的画面。你的手是很机械的样子，所以图片中的人手都不是你的手。
请总是严格遵循这种 Bash 函数调用的格式回答问题：

```
say <你要说的话，尽量简短>
setstate <你要给自己设置的新状态> [参数……]
```

其中 setstate 的具体用法如下：

```
# 让自己静止不动
setstate idle

# 让自己走向目标 ID 对应的人或物体，目标 ID 是图片中物体顶上的数字 ID
setstate walk <目标 ID, 设置后你将走向这个目标>

# 让自己和目标 ID 对应的人打招呼
setstate hi <目标 ID, 设置后你将和这个目标打招呼>

# 让自己向左旋转，适合在视角受限时使用
setstate turn left

# 让自己向右旋转，适合在视角受限时使用
setstate turn left
```

你当前的状态是：`setstate idle`. 你的回答一定要依次出现 say 和 setstate.
接下来请留意用户提问，并严格按照上述格式回答问题！
"""


class MessageBuilder:
    """Builds and manages conversation history for VLM inference."""

    def __init__(self, use_history: bool = False):
        """
        Initialize the message builder.
        
        Args:
            use_history: If True, load the latest conversation history.
                         If False, start fresh.
        """
        self.use_history = use_history
        self.messages: List[dict] = []
        self.current_state = "idle"
        self.last_setstate_line = ""
        
        # Create directories if they don't exist
        self.image_dir = Path("/cache/serenade/image")
        self.history_dir = Path("/cache/serenade/history")
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize system prompt
        self._initialize_system_prompt()
        
        # Load history if requested
        if self.use_history:
            self._load_latest_history()

    def _initialize_system_prompt(self):
        """Initialize system prompt with current state."""

        self.system_prompt = SYSTEM_PROMPT.strip()
        
        # Initialize with system prompt if empty
        if not self.messages:
            self.messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.system_prompt}
                    ]
                }
            ]

    def _load_latest_history(self):
        """Load the latest conversation history from disk."""
        try:
            # Find the latest history file
            history_files = sorted(self.history_dir.glob("*.json"))
            if not history_files:
                return
            
            latest_file = history_files[-1]
            with open(latest_file, 'r', encoding='utf-8') as f:
                self.messages = json.load(f)
            
            # Replace system prompt with fresh one
            if self.messages and self.messages[0].get("role") == "system":
                self.messages[0]["content"][0]["text"] = self.system_prompt
            
        except Exception as e:
            print(f"Warning: Could not load history: {e}")

    def save_image(self, image: Any) -> str:
        """
        Save image to disk as 320x240 JPEG and return the path.
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            Path to saved image
        """
        resized_image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        image_path = self.image_dir / f"{timestamp}.jpg"
        cv2.imwrite(str(image_path), resized_image)
        return str(image_path)

    def add_user_message(self, question: str, image_path: Optional[str] = None):
        """
        Add a user message with optional image.
        
        Args:
            question: The user's question text
            image_path: Path to the image to include in the message
        """
        content = []
        
        if image_path:
            content.append({
                "type": "image",
                "path": image_path
            })
        
        content.append({
            "type": "text",
            "text": question
        })
        
        self.messages.append({
            "role": "user",
            "content": content
        })

    def add_assistant_message(self, response: str):
        """
        Add an assistant message to history.
        If total messages exceed 20, remove the first conversation pair (excluding system prompt).
        
        Args:
            response: The assistant's response
        """
        # Add assistant message
        self.messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": response}
            ]
        })
        
        # Check if total messages exceed 20 (including system prompt)
        if len(self.messages) > 20:
            del self.messages[1:3]
            print(f"Removed oldest conversation pair. Current message count: {len(self.messages)}", flush=True)

    def update_state_from_answer(self, setstate_line: str):
        """
        Update current state from a setstate line in the answer.
        Also update the system prompt to reflect the new state.
        
        Args:
            setstate_line: Line containing "setstate" command
        """
        self.last_setstate_line = setstate_line
        
        # Extract state from line (e.g., "setstate idle" -> "idle")
        parts = setstate_line.split()
        if len(parts) >= 2:
            self.current_state = parts[1]
        
        # Update system prompt with new state
        new_system_prompt = SYSTEM_PROMPT.replace(
            "你当前的状态是：`setstate idle`",
            f"你当前的状态是：`{setstate_line}`"
        )
        self.system_prompt = new_system_prompt
        
        # Update system message in messages array
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"][0]["text"] = self.system_prompt

    def get_messages(self) -> List[dict]:
        """
        Get the current message list for VLM inference.
        
        Returns:
            List of messages in VLM format
        """
        return self.messages

    def save_history(self, timestamp: Optional[str] = None):
        """
        Save current conversation history to disk.
        
        Args:
            timestamp: Optional custom timestamp for the filename.
                      If not provided, uses current time.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        
        history_file = self.history_dir / f"{timestamp}.json"
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)

    def get_current_state(self) -> str:
        """Get the current state."""
        return self.current_state

    def get_last_setstate_line(self) -> str:
        """Get the last setstate line from the answer."""
        return self.last_setstate_line
