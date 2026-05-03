import json
import logging
import os
import re
from typing import Dict, Optional


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _next_run_id(runs_dir: str) -> str:
    pattern = re.compile(r"run_(\d+)$")
    max_n = 0
    if os.path.isdir(runs_dir):
        for name in os.listdir(runs_dir):
            m = pattern.match(name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return f"run_{max_n + 1:03d}"


def create_run_layout(runs_dir: str = "runs", run_id: Optional[str] = None) -> Dict[str, str]:
    _ensure_dir(runs_dir)
    run_id = run_id or _next_run_id(runs_dir)
    run_dir = os.path.join(runs_dir, run_id)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    samples_dir = os.path.join(run_dir, "samples")
    logs_dir = os.path.join(run_dir, "logs")

    _ensure_dir(run_dir)
    _ensure_dir(checkpoints_dir)
    _ensure_dir(samples_dir)
    _ensure_dir(logs_dir)

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "checkpoints_dir": checkpoints_dir,
        "samples_dir": samples_dir,
        "logs_dir": logs_dir,
        "config_path": os.path.join(run_dir, "config.json"),
    }


def save_run_config(config: dict, config_path: str) -> None:
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def configure_logging(logs_dir: str, run_id: str) -> logging.Logger:
    logger = logging.getLogger("handwriting_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt=f"%(asctime)s | {run_id} | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    train_log_path = os.path.join(logs_dir, "training.log")
    error_log_path = os.path.join(logs_dir, "error.log")

    info_handler = logging.FileHandler(train_log_path)
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)

    error_handler = logging.FileHandler(error_log_path)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    return logger


def rebind_module_file_logger(module_logger: logging.Logger, file_path: str, level: int) -> None:
    for handler in list(module_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            module_logger.removeHandler(handler)
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    module_logger.addHandler(file_handler)
