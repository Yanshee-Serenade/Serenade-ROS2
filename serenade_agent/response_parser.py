#!/usr/bin/env python3
"""
Response parser for VLM streamed output.
Parses say and setstate commands from streamed output.
Non-strict parsing to handle noise and partial responses.
Publishes to ROS2 topics as commands are detected.
"""

import re
from serenade_agent.config import TARGET_LEFT, TARGET_RIGHT, TARGET_NONE
from typing import Any, Optional


class ResponseParser:
    """Parses VLM streamed responses into structured commands and publishes them."""

    def __init__(self, logger: Any, answer_publisher: Any, target_publisher: Any, message_builder: Any):
        """
        Initialize the response parser.
        
        Args:
            logger: ROS2 logger for logging
            answer_publisher: ROS2 publisher for /answer topic
            target_publisher: ROS2 publisher for /target topic
            message_builder: MessageBuilder instance to update state
        """
        self.logger = logger
        self.answer_publisher = answer_publisher
        self.target_publisher = target_publisher
        self.message_builder = message_builder
        
        self.buffer = ""
        self.say_content = ""
        self.setstate_line = ""
        self.has_say = False
        self.has_setstate = False

    def parse_streamed_response(self, text_stream):
        """
        Parse a full streamed response from an iterator.
        Processes each line immediately and publishes to topics.
        
        Args:
            text_stream: Iterator of text chunks from the streamer
        """
        self.buffer = ""

        for chunk in text_stream:
            if not chunk:
                continue

            # Add chunk to buffer
            self.buffer += chunk

            # Process complete lines (lines ending with newline)
            while '\n' in self.buffer:
                line, self.buffer = self.buffer.split('\n', 1)
                self._process_line(line)

        # Process any remaining content in buffer
        if self.buffer.strip():
            self._process_line(self.buffer)

    def _process_line(self, line: str):
        """
        Process a single line from the streamed response.
        Publishes and logs immediately.
        
        Args:
            line: A line of text from the streamed response
        """
        line = line.strip()
        if not line:
            return

        # Parse "say" command
        elif "say" in line.lower():
            # Extract content after "say"
            match = re.search(r'say\s+(.+)', line, re.IGNORECASE)
            if match:
                self.say_content = match.group(1).strip()
                self.has_say = True
                # Publish to /answer topic
                from std_msgs.msg import String
                answer_msg = String()
                answer_msg.data = self.say_content
                self.answer_publisher.publish(answer_msg)
                self.logger.info(f"💬 Say: {self.say_content}")

        # Parse "setstate" command
        elif "setstate" in line.lower():
            # Extract the setstate command and arguments
            # Non-strict: accept any content that contains setstate
            match = re.search(r'(setstate\s+\S+(?:\s+\S+)?)', line, re.IGNORECASE)
            if match:
                self.setstate_line = match.group(1).strip()
                self.has_setstate = True
                self.logger.info(f"⚙️ Setstate: {self.setstate_line}")
                
                # Update message builder state
                self.message_builder.update_state_from_answer(self.setstate_line)
                
                # Publish target command
                self._publish_target_command(self.setstate_line)

    def _publish_target_command(self, setstate_line: str):
        """
        Publish target command based on setstate line.
        
        Args:
            setstate_line: The setstate command line
        """
        from std_msgs.msg import String
        
        action = self.get_setstate_action(setstate_line)

        if action == "idle":
            target_msg = String()
            target_msg.data = TARGET_NONE
            self.target_publisher.publish(target_msg)
        elif action == "walk" or action == "hi":
            target_id = self.extract_target_id(setstate_line)
            if target_id:
                target_msg = String()
                target_msg.data = target_id
                self.target_publisher.publish(target_msg)
        elif action == "turn":
            target_id = self.extract_target_id(setstate_line)
            if target_id and target_id.lower() == "left":
                target_msg = String()
                target_msg.data = TARGET_LEFT
                self.target_publisher.publish(target_msg)
            if target_id and target_id.lower() == "right":
                target_msg = String()
                target_msg.data = TARGET_RIGHT
                self.target_publisher.publish(target_msg)

    def get_say(self) -> str:
        """Get accumulated say content."""
        return self.say_content

    def get_setstate(self) -> str:
        """Get setstate line."""
        return self.setstate_line

    def has_all_commands(self) -> bool:
        """Check if all three commands (say, setstate) have been received."""
        return self.has_say and self.has_setstate

    @staticmethod
    def extract_target_id(setstate_line: str) -> Optional[str]:
        """
        Extract target ID from setstate line for walk/hi commands.
        
        Args:
            setstate_line: The setstate command line (e.g., "setstate walk 0")
            
        Returns:
            Target ID if found, None otherwise
        """
        if not setstate_line:
            return None

        parts = setstate_line.split()
        if len(parts) >= 3:
            return parts[2]
        return None

    @staticmethod
    def get_setstate_action(setstate_line: str) -> Optional[str]:
        """
        Extract action from setstate line.
        
        Args:
            setstate_line: The setstate command line
            
        Returns:
            Action (e.g., "idle", "walk", "hi") if found, None otherwise
        """
        if not setstate_line:
            return None

        parts = setstate_line.split()
        if len(parts) >= 2:
            return parts[1]
        return None
