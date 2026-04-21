import pytest

from methods.utils.metafeatures.simple_metric_hook import SimpleEgoMetricsHook
from methods.utils.metafeatures.step_info import StepInfo


def make_step(*, reward: float = 0.0, truncated: bool = False, info: dict | None = None) -> StepInfo:
    return StepInfo(
        observation=[],
        action=0,
        reward=reward,
        next_observation=[],
        terminated=False,
        truncated=truncated,
        info=info or {},
    )


def test_close_collision_rate_counts_ttc_and_crashes() -> None:
    hook = SimpleEgoMetricsHook(close_collision_ttc_threshold=2.0)
    hook.on_probe_start()
    hook.on_episode_start()

    hook.on_step(make_step(reward=1.0, info={"speed": 10.0, "min_ttc": 1.5}))
    hook.on_step(make_step(reward=1.0, info={"speed": 12.0, "min_ttc": 4.0}))
    hook.on_step(make_step(reward=1.0, info={"speed": 0.0, "crashed": True}))

    hook.on_episode_end()

    metrics = hook.finalize()

    assert metrics["close_collision_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["collision_rate"] == pytest.approx(1.0)
