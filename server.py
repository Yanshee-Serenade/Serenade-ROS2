"""
Main server file for Yanshee model server.
This is a modularized version that uses separate modules for different concerns.
"""

import os
import sys

# Add the modules directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "modules"))

from modules.main import main

if __name__ == "__main__":
    main()
