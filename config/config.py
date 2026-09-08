"""
Configuration loader for Episense project.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import os

_CONFIG_CACHE = None
_CONFIG_MTIME = None
_CONFIG_CACHED_PATH = None

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    global _CONFIG_CACHE, _CONFIG_MTIME, _CONFIG_CACHED_PATH
    
    if config_path is None:
        # Find config.yaml in project
        current_dir = Path(__file__).parent
        while current_dir != current_dir.parent:
            config_file = current_dir / "config" / "config.yaml"
            if config_file.exists():
                config_path = str(config_file)
                break
            current_dir = current_dir.parent
        
        if config_path is None:
            # Default to relative path
            config_path = "config/config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        current_mtime = config_path.stat().st_mtime
    except Exception:
        current_mtime = None

    # cache invalidation por mtime: recarrega se arquivo mudou
    if _CONFIG_CACHE is not None and _CONFIG_CACHED_PATH == str(config_path) and _CONFIG_MTIME == current_mtime:
        return _CONFIG_CACHE

    with open(config_path, 'r') as f:
        _CONFIG_CACHE = yaml.safe_load(f)
    _CONFIG_MTIME = current_mtime
    _CONFIG_CACHED_PATH = str(config_path)

    return _CONFIG_CACHE


def get_config(key: str, default=None):
    """Get a specific config value by dot-separated key."""
    config = load_config()
    keys = key.split('.')
    value = config
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k)
        else:
            return default
        if value is None:
            return default
    return value


if __name__ == "__main__":
    config = load_config()
    print(yaml.dump(config, default_flow_style=False))