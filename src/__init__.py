import os
import sys
from pathlib import Path

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory to the directory of the running file
os.chdir(current_dir)