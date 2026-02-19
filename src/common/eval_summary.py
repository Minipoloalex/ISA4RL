import numpy as np
from typing import Dict, Any, Iterable, Union

def aggregate_metrics(
    per_episode: Iterable[Dict[str, Any]],
) -> Dict[str, Union[float, int]]:
    rewards = np.array([entry["reward"] for entry in per_episode], dtype=np.float64)
    lengths = np.array([entry["length"] for entry in per_episode], dtype=np.int32)
    summary: Dict[str, Union[float, int]] = {
        "episodes": len(rewards),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
    }
    bool_fields = {"crashed", "is_success", "success"}
    for field in bool_fields:
        values = [entry.get(field) for entry in per_episode if field in entry]
        if values:
            summary[f"{field}_rate"] = float(np.mean(values))   # type: ignore
    return summary
