import os
import json
from typing import List, Dict


def read_prompt(path: str | os.PathLike) -> Dict:
    """Reads from JSON-file"""
    with open(path, "r") as f:
        return json.load(f)
