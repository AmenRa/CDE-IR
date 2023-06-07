import os
from pathlib import Path


def base_path():
    p = Path(os.environ.get("CDE_IR_BASE_PATH"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def datasets_path():
    p = base_path() / "datasets"
    p.mkdir(parents=True, exist_ok=True)
    return p
