"""
Common functionality for RhoFold MCP scripts.

This module contains logging setup and RhoFold module loading functions
that are shared across multiple scripts.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any


def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration for RhoFold scripts.

    Args:
        output_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('RhoFold_MCP')
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers to avoid duplication
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

    # File handler
    file_handler = logging.FileHandler(output_dir / 'log.txt', mode='w')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(getattr(logging, log_level.upper()))
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def load_rhofold_modules() -> Dict[str, Any]:
    """
    Lazy load RhoFold modules to minimize startup time.

    Returns:
        Dict containing loaded RhoFold modules and functions

    Raises:
        ImportError: If RhoFold modules cannot be imported
    """
    try:
        # Add repo path to Python path for imports
        script_dir = Path(__file__).parent.parent
        repo_path = script_dir / "repo" / "RhoFold"
        sys.path.insert(0, str(repo_path))

        from rhofold.rhofold import RhoFold
        from rhofold.config import rhofold_config
        from rhofold.utils import get_device, timing, save_ss2ct
        from rhofold.utils.alphabet import get_features
        from rhofold.relax.relax import AmberRelaxation

        import torch
        import numpy as np
        from huggingface_hub import snapshot_download

        return {
            'RhoFold': RhoFold,
            'rhofold_config': rhofold_config,
            'get_device': get_device,
            'timing': timing,
            'get_features': get_features,
            'AmberRelaxation': AmberRelaxation,
            'save_ss2ct': save_ss2ct,
            'torch': torch,
            'np': np,
            'snapshot_download': snapshot_download
        }

    except ImportError as e:
        raise ImportError(
            f"Failed to import RhoFold modules: {e}\n"
            "Make sure you're running this from the RhoFold conda environment (env_py37)\n"
            "Activate with: mamba run -p ./env_py37 python <script> ..."
        )