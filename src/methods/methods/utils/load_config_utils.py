from methods.configs import TrainConfig, InstanceConfig
from common.file_utils import *

def load_env_instance_configs(env_name: str, base_results_path: Path = BASE_RESULTS_PATH) -> List[InstanceConfig]:
    configs = read_json(TRAIN_CONFIGS_PATH(env_name))
    return [InstanceConfig.from_dict(env_name, config, base_results_path) for config in configs]

def load_env_train_configs(env_name: str, base_results_path: Path = BASE_RESULTS_PATH) -> List[TrainConfig]:
    configs = read_json(TRAIN_CONFIGS_PATH(env_name))
    return [TrainConfig.from_dict(env_name, config, base_results_path, use_best_model=True) for config in configs]

def is_trained(config: TrainConfig) -> bool:
    """Trained iff training metadata artifact exists."""
    return nonempty_file_in(RESULTS_TRAIN_METADATA_PATH(config.train_folder_path))

def is_evaluated(config: TrainConfig) -> bool:
    """Evaluated iff evaluation results artifact exists."""
    return nonempty_file_in(RESULTS_EVALUATION_PATH(config.train_folder_path))

def is_extracted(config: InstanceConfig) -> bool:
    """Extracted iff metafeatures result artifact exists."""
    return nonempty_file_in(RESULTS_METAFEATURES_PATH(config.instance_folder_path))

