from methods.main import main as methods_main

from carla_agents.gymdrive_adapter import register_carla_env


def main() -> None:
    register_carla_env()
    methods_main(["carla"])


if __name__ == "__main__":
    main()
