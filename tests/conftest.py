"""Tests that need no GPU, no network and no API key: `pytest` from the repo root, a few seconds."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
