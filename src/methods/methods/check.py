from typing import List, Callable
from methods.configs import TrainConfig, InstanceConfig

ConfigBoolFunc = Callable[[TrainConfig], bool] | Callable[[InstanceConfig], bool]

def check_helper(configs: List[InstanceConfig] | List[TrainConfig], checker_func: ConfigBoolFunc):
    segs = []   # segments of not checker_func configurations (e.g., not trained configurations)
    i = 0
    while i < len(configs):
        if not checker_func(configs[i]):
            lo = i
            hi = i + 1
            while hi < len(configs) and not checker_func(configs[hi]):
                hi += 1
            segs.append((lo, hi))
            i = hi - 1
        i += 1
    return segs

