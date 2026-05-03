from dataclasses import dataclass
from typing import Any, Dict, Iterable


@dataclass
class ConfigValidationError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def require_keys(d: Dict[str, Any], path: str, keys: Iterable[str]) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ConfigValidationError(f"Missing keys at '{path}': {missing}")
