"""
Shared library for RhoFold MCP scripts.

This module contains common functionality extracted from the MCP scripts
to minimize duplication and provide a clean interface.
"""

from .common import setup_logging, load_rhofold_modules
from .io import load_config, save_json
from .validation import validate_rna_sequence, validate_files

__all__ = [
    'setup_logging',
    'load_rhofold_modules',
    'load_config',
    'save_json',
    'validate_rna_sequence',
    'validate_files'
]