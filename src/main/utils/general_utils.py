import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import torch
from pprint import pprint

from common.file_utils import *
from common.config_utils import *

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def discretize(obs: np.ndarray, clip_range: Tuple[float, float] = (-5.0, 5.0), bins_per_dim: int = 15) -> Tuple[int, ...]:
    obs = np.asarray(obs, dtype=np.float32)
    low, high = clip_range
    obs = np.clip(obs, low, high)
    edges = np.linspace(low, high, bins_per_dim + 1)
    return tuple(np.digitize(obs, edges))


def _flatten_obs(obs: Any) -> np.ndarray:
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32).ravel()
    if isinstance(obs, (list, tuple)):
        return np.asarray(obs, dtype=np.float32).ravel()
    if isinstance(obs, (int, float)):
        return np.asarray([obs], dtype=np.float32)
    raise TypeError(f"Unsupported observation type: {type(obs)}")


def _normalize_action(action: Any) -> Any:
    if isinstance(action, np.ndarray):
        return tuple(action.astype(np.float32).ravel().tolist())
    if isinstance(action, (list, tuple)):
        return tuple(np.asarray(action, dtype=np.float32).ravel().tolist())
    if isinstance(action, (np.integer, int)):  # type: ignore[arg-type]
        return int(action)
    if isinstance(action, (np.floating, float)):  # type: ignore[arg-type]
        return float(action)
    return action


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _interpolate_range_value(lo: Any, hi: Any, weight: float) -> Any:
    if isinstance(lo, (int, np.integer)) and isinstance(hi, (int, np.integer)):  # type: ignore[arg-type]
        raw = float(lo + (hi - lo) * weight)
        clipped = max(lo, min(hi, _round_half_up(raw)))
        return int(clipped)
    raw = float(lo + (hi - lo) * weight)
    return float(round(max(lo, min(hi, raw)), 6))


def _coerce_numeric(value: Any) -> Optional[float]:
    if isinstance(value, (int, float, np.integer, np.floating)):  # type: ignore[attr-defined]
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
