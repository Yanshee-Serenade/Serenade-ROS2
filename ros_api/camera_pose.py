#!/usr/bin/env python3
import json
import socket
import threading
import time

from ros_api import CameraPoseData


class CameraPoseClient:
    def __init__(self, host="localhost", port=51118, scale=67.0):
        self.host = host
        self.port = port
        self.sock = None
        self._shutdown = False
        self._latest_pose = None
        self._latest_timestamp = 0
        self._pose_lock = threading.Lock()
        self._stream_thread = None
        self._scale = scale

    def connect(self):
        """Connects to the server with retries."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Set a timeout so we can exit cleanly if needed
            self.sock.settimeout(None)
            self.sock.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port}")
            return True
        except ConnectionRefusedError:
            print(f"Connection failed: {self.host}:{self.port}")
            return False

    def start_streaming(self):
        """Start streaming poses in a background thread."""
        if self._stream_thread and self._stream_thread.is_alive():
            print("Streaming already started")
            return

        if not self.sock and not self.connect():
            return False

        self._shutdown = False
        self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._stream_thread.start()
        print("Started camera pose streaming thread")
        return True

    def _stream_loop(self):
        """Background thread that continuously reads pose data."""
        if not self.sock:
            return
        f_obj = self.sock.makefile("r", encoding="utf-8")

        try:
            for line in f_obj:
                if self._shutdown:
                    break
                try:
                    data = json.loads(line.strip())
                    # Update latest pose with thread-safe lock
                    with self._pose_lock:
                        self._latest_pose = CameraPoseData.from_dict(data)
                        self._latest_timestamp = time.time()
                except (json.JSONDecodeError, KeyError):
                    continue
        except (ConnectionResetError, BrokenPipeError, OSError):
            print("Stream ended (server disconnected).")
        finally:
            self.close()

    def get_latest_pose(self):
        """
        Get the latest camera pose and timestamp since start.

        Returns:
            tuple: (pose_data, timestamp_seconds) or (None, 0.0) if no pose available
        """
        with self._pose_lock:
            if self._latest_pose is None:
                return None, 0.0
            return self._latest_pose, self._latest_timestamp

    def stream(self):
        """
        Generator yielding (topic_type, data_dict).
        topic_type will be 'raw' or 'estimated'.

        Note: For continuous streaming, use start_streaming() and get_latest_pose()
        instead of this generator.
        """
        if not self.sock and not self.connect():
            return

        # 'makefile' buffers the stream efficiently for line-by-line reading
        if not self.sock:
            return
        f_obj = self.sock.makefile("r", encoding="utf-8")

        try:
            for line in f_obj:
                if self._shutdown:
                    break
                try:
                    data = json.loads(line.strip())
                    # Yield the label and the full data object
                    yield data["topic"], CameraPoseData.from_dict(data)
                except (json.JSONDecodeError, KeyError):
                    continue
        except (ConnectionResetError, BrokenPipeError, OSError):
            print("Stream ended (server disconnected).")
        finally:
            self.close()

    def close(self):
        self._shutdown = True
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=1.0)
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()
            self.sock = None
