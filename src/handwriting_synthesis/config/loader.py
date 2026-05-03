import json
import os
from typing import Any, Dict

from .schema import ConfigValidationError, require_keys


def load_json_config(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise ConfigValidationError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ConfigValidationError("Config root must be a JSON object")
    return cfg


def validate_prepare_config(cfg: Dict[str, Any]) -> None:
    require_keys(cfg, "root", ["dataset", "output"])
    require_keys(cfg["dataset"], "dataset", ["provider_name", "provider_args", "max_len"])
    require_keys(cfg["output"], "output", ["prepared_data_dir"])


def validate_train_config(cfg: Dict[str, Any]) -> None:
    require_keys(cfg, "root", ["dataset", "training", "output"])
    require_keys(cfg["dataset"], "dataset", ["prepared_data_dir", "charset_path"])
    require_keys(
        cfg["training"],
        "training",
        ["batch_size", "epochs", "sampling_interval", "clip1", "clip2", "unconditional", "device"],
    )
    require_keys(cfg["output"], "output", ["model_dir", "samples_dir"])
