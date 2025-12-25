"""
Validation utilities for RhoFold MCP scripts.

This module contains functions for validating RNA sequences,
file formats, and input parameters.
"""

from pathlib import Path
from typing import List, Set


def validate_rna_sequence(fasta_file: Path) -> str:
    """
    Validate RNA sequence contains only valid nucleotides.

    Args:
        fasta_file: Path to FASTA file

    Returns:
        RNA sequence string

    Raises:
        ValueError: If sequence contains invalid nucleotides
    """
    valid_nucleotides: Set[str] = {'A', 'U', 'G', 'C', 'N'}  # N for unknown

    with open(fasta_file, 'r') as f:
        lines = f.readlines()

    sequence = ""
    for line in lines:
        if not line.startswith('>'):
            sequence += line.strip().upper()

    invalid_chars = set(sequence) - valid_nucleotides
    if invalid_chars:
        raise ValueError(f"Invalid nucleotides found: {invalid_chars}. RNA sequences should contain only A, U, G, C.")

    return sequence


def validate_files(*file_paths: Path) -> None:
    """
    Validate that files exist and are readable.

    Args:
        *file_paths: Variable number of file paths to validate

    Raises:
        FileNotFoundError: If any file doesn't exist
    """
    for file_path in file_paths:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")


def validate_fasta_format(fasta_file: Path) -> None:
    """
    Validate FASTA file format.

    Args:
        fasta_file: Path to FASTA file

    Raises:
        ValueError: If file format is invalid
    """
    with open(fasta_file, 'r') as f:
        first_line = f.readline().strip()
        if not first_line.startswith('>'):
            raise ValueError(f"Invalid FASTA format in {fasta_file}")


def validate_a3m_format(a3m_file: Path) -> None:
    """
    Validate A3M (MSA) file format.

    Args:
        a3m_file: Path to A3M file

    Raises:
        ValueError: If file format is invalid
    """
    with open(a3m_file, 'r') as f:
        first_line = f.readline().strip()
        if not first_line.startswith('>'):
            raise ValueError(f"Invalid A3M format in {a3m_file}")


def count_msa_sequences(msa_file: Path) -> int:
    """
    Count number of sequences in MSA file.

    Args:
        msa_file: Path to MSA file

    Returns:
        Number of sequences
    """
    count = 0
    with open(msa_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                count += 1
    return count