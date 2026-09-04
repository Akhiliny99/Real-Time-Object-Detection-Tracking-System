"""Small shared helper so every script loads config.yaml the same way."""
from pathlib import Path
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Copy config.yaml to the project root or pass --config explicitly."
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
