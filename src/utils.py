import importlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import List

import requests
from omegaconf import DictConfig
from tqdm import tqdm
from unified_io import join_path


def download_file(url: str, path: str, desc: str = "Downloading", resume: bool = True):
    path = Path(path)
    byte_position = path.stat().st_size if resume and path.exists() else 0

    # Get file size
    file_size = int(urllib.request.urlopen(url).info().get_all("Content-Length")[0])

    if byte_position >= file_size:
        return

    # Add information to resume download at specific byte position to header
    resume_header = {"Range": f"bytes={byte_position}-"} if byte_position > 0 else None

    # Establish connection
    r = requests.get(url, stream=True, headers=resume_header)

    block_size = 1024

    with open(path, mode="ab" if byte_position > 0 else "wb") as f:
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=desc,
            initial=byte_position,
            miniters=1,
            dynamic_ncols=True,
        ) as pbar:
            for chunk in r.iter_content(32 * block_size):
                f.write(chunk)
                pbar.update(len(chunk))


def load_obj(obj_path: str, default_obj_path: str = ""):
    """Extract an object from a given path. Taken from: https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py

    Args:
        obj_path (str): Path to an object to be extracted, including the object name.
        default_obj_path (str, optional): Default object path.

    Returns:
        Extracted object.

    Raises:
        AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            "Object `{}` cannot be loaded from `{}`.".format(obj_name, obj_path)
        )
    return getattr(module_obj, obj_name)


def setup_logger(logger, dir: str, filename: str):
    logger.remove()
    logger.add(sys.stderr, level="TRACE")

    # Remove log file if exists
    try:
        os.remove(join_path(dir, filename))
    except:
        pass

    # Define logger output behaviour
    logger.add(
        join_path(dir, filename),
        enqueue=True,
        backtrace=True,
        diagnose=True,
        level="TRACE",
    )


def get_pl_loggers(cfg: DictConfig) -> List:
    loggers = []

    if cfg.logging.do_logs:
        loggers.extend(
            load_obj(logger.class_name)(**logger.params)
            for logger in cfg.logging.pl_loggers
        )

    return loggers


def get_pl_callbacks(cfg: DictConfig) -> List:
    return [
        load_obj(callback.class_name)(**callback.params)
        for callback in cfg.callbacks.values()
        if callback.use
    ]
