"""
I/O utilities for RhoFold MCP scripts.

This module contains functions for loading configuration files,
reading/writing data, and file management.
"""

import json
from pathlib import Path
from typing import Dict, Any, Union


def load_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        config_file: Path to JSON configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save
        file_path: Path to save file
        indent: JSON indentation level
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def ensure_directory(dir_path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating if necessary.

    Args:
        dir_path: Directory path

    Returns:
        Path object for the directory
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path