"""
Configuration management utilities.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary containing configuration parameters.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Compute train cases automatically
    if "data" in config:
        all_cases = set(range(140))
        test_cases = set(config["data"].get("test_cases", []))
        val_cases = set(config["data"].get("val_cases", []))
        train_cases = sorted(all_cases - test_cases - val_cases)
        config["data"]["train_cases"] = train_cases

    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to a YAML file.

    Args:
        config: Configuration dictionary.
        save_path: Path to save the YAML file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def update_config(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update config with overrides.

    Args:
        config: Base configuration dictionary.
        overrides: Override values.

    Returns:
        Updated configuration dictionary.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            config[key] = update_config(config[key], value)
        else:
            config[key] = value
    return config
