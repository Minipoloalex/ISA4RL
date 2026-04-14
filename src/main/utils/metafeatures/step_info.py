from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, SupportsFloat

import numpy as np

@dataclass(frozen=True)
class StepInfo:
    """Represents a single transition in a highway-env episode."""

    observation: np.ndarray
    action: Any
    reward: SupportsFloat
    next_observation: np.ndarray
    terminated: bool
    truncated: bool

    # Extra metadata
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Useful for converting to a Pandas DataFrame or JSON."""
        # As an example for now
        return {
            "observation": self.observation.tolist(),
            "action": self.action,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }
