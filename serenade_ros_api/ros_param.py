#!/usr/bin/env python3
import json
import socket


class ROSParamClient:
    def __init__(self, host="localhost", port=21119):
        self.host = host
        self.port = port

    def set_param(self, path, value):
        """
        Connects to the server, sets the param, and closes the connection.

        :param path: The ROS parameter path (e.g., '/my_node/gain')
        :param value: The value to set (int, float, string, list, bool, etc.)
        """
        payload = {"path": path, "value": value}

        # Convert to JSON and add a newline as a delimiter
        message = json.dumps(payload) + "\n"

        try:
            with socket.create_connection((self.host, self.port), timeout=5) as sock:
                sock.sendall(message.encode("utf-8"))
        except ConnectionRefusedError:
            print(f"Error: Could not connect to {self.host}:{self.port}")
        except Exception as e:
            print(f"Error sending param: {e}")
