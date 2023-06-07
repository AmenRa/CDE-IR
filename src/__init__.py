import os
from pathlib import Path

# Set environment variables ----------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CDE_IR_BASE_PATH"] = str(Path.home() / ".cde_ir")


def set_base_path(path: str):
    os.environ["CDE_IR_BASE_PATH"] = path
