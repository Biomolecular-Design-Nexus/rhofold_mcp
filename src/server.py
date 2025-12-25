#!/usr/bin/env python3
"""
MCP Server for RhoFold

Provides both synchronous and asynchronous (submit) APIs for RNA structure prediction,
analysis, and batch processing using RhoFold+.

This server wraps the clean scripts from the scripts/ directory and provides:
- Sync tools for fast operations (< 10 minutes)
- Submit tools for long-running operations (> 10 minutes)
- Job management for background processing
"""

import sys
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
import logging

# Add project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

# Import FastMCP after path setup
try:
    from fastmcp import FastMCP
except ImportError:
    print("FastMCP not installed. Please run: pip install fastmcp")
    sys.exit(1)

# Import job manager
from jobs.manager import job_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server
mcp = FastMCP("RhoFold")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, and any errors
    """
    return job_manager.get_job_status(job_id)

@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or error if not completed
    """
    return job_manager.get_job_result(job_id)

@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)

@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)

@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status
    """
    return job_manager.list_jobs(status)

# ==============================================================================
# Synchronous Tools (for fast operations < 10 min)
# ==============================================================================

@mcp.tool()
def predict_rna_structure(
    input_file: str,
    output_file: Optional[str] = None,
    device: str = "cpu",
    relax_steps: int = 1000
) -> dict:
    """
    Predict RNA 3D structure from a single sequence using RhoFold.

    Fast operation suitable for single sequences (~5 minutes).
    For batch processing of multiple sequences, use submit_batch_prediction.

    Args:
        input_file: Path to RNA sequence in FASTA format
        output_file: Optional path to save output directory
        device: Device for computation (cpu, cuda:0, etc.)
        relax_steps: Number of Amber relaxation steps (default: 1000)

    Returns:
        Dictionary with prediction results and output paths

    Example:
        predict_rna_structure("examples/data/3owzA/3owzA.fasta", "results/3owzA_output")
    """
    try:
        from rna_single_sequence_prediction import run_single_sequence_prediction

        result = run_single_sequence_prediction(
            input_file=input_file,
            output_file=output_file,
            device=device,
            relax_steps=relax_steps
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ImportError as e:
        return {
            "status": "error",
            "error": f"RhoFold modules not available: {e}. Ensure you're using the env_py37 environment."
        }
    except Exception as e:
        logger.error(f"RNA structure prediction failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def predict_rna_structure_with_msa(
    fasta_file: str,
    msa_file: str,
    output_file: Optional[str] = None,
    device: str = "cpu",
    relax_steps: int = 1000
) -> dict:
    """
    Predict RNA 3D structure using Multiple Sequence Alignment with RhoFold.

    Enhanced prediction using evolutionary information from MSA.
    Fast operation suitable for single sequences (~5 minutes).

    Args:
        fasta_file: Path to RNA sequence in FASTA format
        msa_file: Path to MSA file in A3M format
        output_file: Optional path to save output directory
        device: Device for computation (cpu, cuda:0, etc.)
        relax_steps: Number of Amber relaxation steps (default: 1000)

    Returns:
        Dictionary with prediction results and output paths

    Example:
        predict_rna_structure_with_msa(
            "examples/data/3owzA/3owzA.fasta",
            "examples/data/3owzA/3owzA.a3m",
            "results/3owzA_msa_output"
        )
    """
    try:
        from rna_msa_prediction import run_msa_prediction

        result = run_msa_prediction(
            fasta_file=fasta_file,
            msa_file=msa_file,
            output_file=output_file,
            device=device,
            relax_steps=relax_steps
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ImportError as e:
        return {
            "status": "error",
            "error": f"RhoFold modules not available: {e}. Ensure you're using the env_py37 environment."
        }
    except Exception as e:
        logger.error(f"MSA-based RNA structure prediction failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def analyze_rna_structures(
    input_dir: str,
    output_file: Optional[str] = None,
    export_csv: bool = True,
    include_unrelaxed: bool = False
) -> dict:
    """
    Analyze and validate RNA structures from RhoFold outputs.

    Fast operation suitable for analyzing multiple structures (~10 seconds).
    Provides comprehensive analysis of structure quality and validation.

    Args:
        input_dir: Path to directory containing RhoFold prediction results
        output_file: Optional path to save analysis results
        export_csv: Whether to export results as CSV format
        include_unrelaxed: Whether to analyze unrelaxed structures too

    Returns:
        Dictionary with analysis results and statistics

    Example:
        analyze_rna_structures("output/batch_results", "analysis/batch_analysis")
    """
    try:
        from rna_structure_analysis import run_structure_analysis

        result = run_structure_analysis(
            input_dir=input_dir,
            output_file=output_file,
            export_csv=export_csv,
            include_unrelaxed=include_unrelaxed
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"Input directory not found: {e}"}
    except Exception as e:
        logger.error(f"RNA structure analysis failed: {e}")
        return {"status": "error", "error": str(e)}

# ==============================================================================
# Submit Tools (for long-running operations > 10 min)
# ==============================================================================

@mcp.tool()
def submit_batch_rna_prediction(
    input_dir: Optional[str] = None,
    multi_fasta: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_sequences: Optional[int] = None,
    device: str = "cpu",
    relax_steps: int = 1000,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch RNA structure prediction for background processing.

    This operation processes multiple RNA sequences and may take more than 10 minutes.
    Use get_job_status() to monitor progress and get_job_result() to retrieve results.

    Args:
        input_dir: Path to directory containing FASTA files (and optional MSA files)
        multi_fasta: Path to multi-FASTA file (alternative to input_dir)
        output_dir: Directory to save outputs
        max_sequences: Maximum number of sequences to process (safety limit)
        device: Device for computation (cpu, cuda:0, etc.)
        relax_steps: Number of Amber relaxation steps per structure
        job_name: Optional name for the job (for easier tracking)

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see execution logs

    Example:
        submit_batch_rna_prediction("examples/data", output_dir="results/batch", max_sequences=5)
    """
    script_path = str(SCRIPTS_DIR / "rna_batch_prediction.py")

    args = {
        "device": device,
        "relax_steps": relax_steps
    }

    if input_dir:
        args["input_dir"] = input_dir
    elif multi_fasta:
        args["multi_fasta"] = multi_fasta
    else:
        return {
            "status": "error",
            "error": "Either input_dir or multi_fasta must be provided"
        }

    if output_dir:
        args["output"] = output_dir
    if max_sequences:
        args["max_sequences"] = max_sequences

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"batch_rna_prediction_{max_sequences or 'all'}_sequences"
    )

@mcp.tool()
def submit_large_sequence_prediction(
    input_file: str,
    output_dir: Optional[str] = None,
    device: str = "cpu",
    relax_steps: int = 1000,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit single sequence prediction for very large sequences (background processing).

    This is useful for very large RNA sequences that may take more than 10 minutes
    or when you want to run predictions in the background.

    Args:
        input_file: Path to RNA sequence in FASTA format
        output_dir: Directory to save outputs
        device: Device for computation (cpu, cuda:0, etc.)
        relax_steps: Number of Amber relaxation steps
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the prediction job

    Example:
        submit_large_sequence_prediction("large_rna.fasta", "results/large_prediction")
    """
    script_path = str(SCRIPTS_DIR / "rna_single_sequence_prediction.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input": input_file,
            "output": output_dir,
            "device": device,
            "relax_steps": relax_steps
        },
        job_name=job_name or f"large_sequence_prediction"
    )

@mcp.tool()
def submit_msa_prediction(
    fasta_file: str,
    msa_file: str,
    output_dir: Optional[str] = None,
    device: str = "cpu",
    relax_steps: int = 1000,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit MSA-based prediction for background processing.

    This is useful for large MSAs or when you want to run predictions in the background.

    Args:
        fasta_file: Path to RNA sequence in FASTA format
        msa_file: Path to MSA file in A3M format
        output_dir: Directory to save outputs
        device: Device for computation (cpu, cuda:0, etc.)
        relax_steps: Number of Amber relaxation steps
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the MSA prediction job

    Example:
        submit_msa_prediction("seq.fasta", "seq.a3m", "results/msa_prediction")
    """
    script_path = str(SCRIPTS_DIR / "rna_msa_prediction.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "fasta": fasta_file,
            "msa": msa_file,
            "output": output_dir,
            "device": device,
            "relax_steps": relax_steps
        },
        job_name=job_name or "msa_based_prediction"
    )

@mcp.tool()
def submit_comprehensive_analysis(
    prediction_results_dir: str,
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit comprehensive structure analysis for background processing.

    This is useful for analyzing large numbers of predicted structures.

    Args:
        prediction_results_dir: Directory containing RhoFold prediction results
        output_dir: Directory to save analysis outputs
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the analysis job

    Example:
        submit_comprehensive_analysis("results/batch_predictions", "analysis/comprehensive")
    """
    script_path = str(SCRIPTS_DIR / "rna_structure_analysis.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input": prediction_results_dir,
            "output": output_dir,
            "export_csv": True
        },
        job_name=job_name or "comprehensive_structure_analysis"
    )

# ==============================================================================
# Utility Tools
# ==============================================================================

@mcp.tool()
def validate_rna_sequence(sequence_file: str) -> dict:
    """
    Validate RNA sequence file format and contents.

    Args:
        sequence_file: Path to FASTA file to validate

    Returns:
        Dictionary with validation results and sequence information
    """
    try:
        from pathlib import Path
        input_file = Path(sequence_file)

        if not input_file.exists():
            return {"status": "error", "error": f"File not found: {sequence_file}"}

        # Basic FASTA validation
        with open(input_file, 'r') as f:
            lines = f.readlines()

        if not lines or not lines[0].startswith('>'):
            return {"status": "error", "error": "Invalid FASTA format: missing header"}

        # Extract sequence
        sequence = ""
        for line in lines[1:]:
            if not line.startswith('>'):
                sequence += line.strip().upper()

        # Validate nucleotides
        valid_nucleotides = set(['A', 'U', 'G', 'C', 'N'])
        invalid_chars = set(sequence) - valid_nucleotides

        if invalid_chars:
            return {
                "status": "error",
                "error": f"Invalid nucleotides found: {invalid_chars}. RNA sequences should contain only A, U, G, C."
            }

        return {
            "status": "success",
            "sequence_length": len(sequence),
            "sequence_preview": sequence[:50] + ("..." if len(sequence) > 50 else ""),
            "valid": True
        }

    except Exception as e:
        return {"status": "error", "error": f"Validation failed: {e}"}

@mcp.tool()
def get_example_data_info() -> dict:
    """
    Get information about available example datasets for testing.

    Returns:
        Dictionary with example files and their descriptions
    """
    try:
        examples_dir = MCP_ROOT / "examples" / "data"
        if not examples_dir.exists():
            return {"status": "error", "error": "Examples directory not found"}

        examples = []
        for item in examples_dir.iterdir():
            if item.is_dir():
                fasta_file = item / f"{item.name}.fasta"
                msa_file = item / f"{item.name}.a3m"

                example_info = {
                    "name": item.name,
                    "directory": str(item),
                    "has_fasta": fasta_file.exists(),
                    "has_msa": msa_file.exists()
                }

                if fasta_file.exists():
                    example_info["fasta_file"] = str(fasta_file)
                    # Get sequence length
                    try:
                        with open(fasta_file, 'r') as f:
                            seq = "".join(line.strip() for line in f if not line.startswith('>'))
                        example_info["sequence_length"] = len(seq)
                    except:
                        pass

                if msa_file.exists():
                    example_info["msa_file"] = str(msa_file)

                examples.append(example_info)

        return {
            "status": "success",
            "examples_directory": str(examples_dir),
            "examples": examples,
            "total_examples": len(examples)
        }

    except Exception as e:
        return {"status": "error", "error": f"Failed to get example data info: {e}"}

# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    logger.info("Starting RhoFold MCP Server")
    logger.info(f"Scripts directory: {SCRIPTS_DIR}")
    logger.info(f"Jobs will be stored in: {job_manager.jobs_dir}")
    mcp.run()