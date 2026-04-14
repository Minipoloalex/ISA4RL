from typing import Protocol, Dict

from .step_info import StepInfo

class BaseMetricHook(Protocol):
    def on_probe_start(self) -> None:
        ...

    def on_episode_start(self) -> None:
        ...

    def on_step(self, context: StepInfo) -> None:
        ...

    def on_episode_end(self) -> None:
        ...

    def finalize(self) -> Dict[str, float]:
        ...
