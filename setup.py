import os
from pathlib import Path

Path("logs").mkdir(parents=True, exist_ok=True)
Path("dataset").mkdir(parents=True, exist_ok=True)
Path("weights").mkdir(parents=True, exist_ok=True)
Path("results").mkdir(parents=True, exist_ok=True)
Path("assets").mkdir(parents=True, exist_ok=True)
