from pathlib import Path

Path("logs").mkdir(parents=True, exist_ok=True)
Path("replay_buffer_storage").mkdir(parents=True, exist_ok=True)
Path("weights").mkdir(parents=True, exist_ok=True)
Path("results").mkdir(parents=True, exist_ok=True)
