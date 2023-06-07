import importlib
import os
import sys
from typing import List

from omegaconf import DictConfig
from unified_io import join_path


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
