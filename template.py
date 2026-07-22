import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    ".github/workflows/.gitkeep",
    "src/Heart/__init__.py",
    "src/Heart/components/__init__.py",
    "src/Heart/utils/__init__.py",
    "src/Heart/pipeline/__init__.py",
]
