#!/usr/bin/env python3
"""
Script: rna_msa_prediction.py
Description: Clean RNA 3D structure prediction with Multiple Sequence Alignment

Original Use Case: examples/use_case_2_prediction_with_msa.py
Dependencies Removed: Inlined logging setup, validation, repo path management

Usage:
    python scripts/rna_msa_prediction.py --fasta <fasta_file> --msa <msa_file> --output <output_file>

Example:
    python scripts/rna_msa_prediction.py --fasta examples/data/3owzA/3owzA.fasta --msa examples/data/3owzA/3owzA.a3m --output results/3owzA_msa
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import os
import sys
import logging
import json
from pathlib import Path
from typing import Union, Optional, Dict, Any

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "device": "cpu",
    "checkpoint_path": "./pretrained/rhofold_pretrained_params.pt",
    "relax_steps": 1000,
    "log_level": "INFO"
}

# ==============================================================================
# Inlined Utility Functions (simplified from repo)
# ==============================================================================
def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration. Inlined from original use case."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('RhoFold_MSA')
    logger.setLevel(getattr(logging, log_level.upper()))

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

def validate_files(fasta_file: Path, msa_file: Path) -> None:
    """Validate input files exist and are readable. Inlined from original."""
    if not fasta_file.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_file}")

    if not msa_file.exists():
        raise FileNotFoundError(f"MSA file not found: {msa_file}")

    # Check FASTA format
    with open(fasta_file, 'r') as f:
        first_line = f.readline().strip()
        if not first_line.startswith('>'):
            raise ValueError(f"Invalid FASTA format in {fasta_file}")

    # Check MSA format (basic A3M validation)
    with open(msa_file, 'r') as f:
        first_line = f.readline().strip()
        if not first_line.startswith('>'):
            raise ValueError(f"Invalid A3M format in {msa_file}")

def count_msa_sequences(msa_file: Path) -> int:
    """Count number of sequences in MSA file. Inlined from original."""
    count = 0
    with open(msa_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                count += 1
    return count

def load_rhofold_modules():
    """Lazy load RhoFold modules to minimize startup time."""
    try:
        # Add repo path to Python path for imports
        script_dir = Path(__file__).parent
        repo_path = script_dir.parent / "repo" / "RhoFold"
        sys.path.insert(0, str(repo_path))

        from rhofold.rhofold import RhoFold
        from rhofold.config import rhofold_config
        from rhofold.utils import get_device, timing
        from rhofold.utils.alphabet import get_features
        from rhofold.relax.relax import AmberRelaxation
        from rhofold.utils import save_ss2ct

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
            "Activate with: mamba run -p ./env_py37 python scripts/rna_msa_prediction.py ..."
        )

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_msa_prediction(
    fasta_file: Union[str, Path],
    msa_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for RNA 3D structure prediction with MSA.

    Args:
        fasta_file: Path to RNA sequence in FASTA format
        msa_file: Path to Multiple Sequence Alignment in A3M format
        output_file: Path to save output directory (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - result: Main computation result paths
            - output_dir: Path to output directory
            - metadata: Execution metadata

    Example:
        >>> result = run_msa_prediction("input.fasta", "input.a3m", "output_dir")
        >>> print(result['output_dir'])
    """
    # Setup
    fasta_file = Path(fasta_file)
    msa_file = Path(msa_file)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Setup output directory
    if output_file:
        output_dir = Path(output_file)
    else:
        output_dir = Path("output") / "msa_prediction"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir, config.get('log_level', 'INFO'))

    logger.info("=" * 70)
    logger.info("RhoFold+ RNA 3D Structure Prediction with MSA")
    logger.info("=" * 70)
    logger.info(f"Input FASTA: {fasta_file}")
    logger.info(f"Input MSA: {msa_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {config['device']}")
    logger.info(f"Checkpoint: {config['checkpoint_path']}")
    logger.info(f"Amber relaxation steps: {config['relax_steps']}")

    # Validate input files
    validate_files(fasta_file, msa_file)
    msa_count = count_msa_sequences(msa_file)
    logger.info(f"Input files validated successfully")
    logger.info(f"MSA contains {msa_count} sequences")

    # Load RhoFold modules
    modules = load_rhofold_modules()

    # Extract modules for easier access
    RhoFold = modules['RhoFold']
    rhofold_config = modules['rhofold_config']
    get_device = modules['get_device']
    timing = modules['timing']
    get_features = modules['get_features']
    AmberRelaxation = modules['AmberRelaxation']
    save_ss2ct = modules['save_ss2ct']
    torch = modules['torch']
    np = modules['np']
    snapshot_download = modules['snapshot_download']

    try:
        # Load model
        logger.info("Loading RhoFold model...")
        model = RhoFold(rhofold_config)

        # Download checkpoint if needed
        checkpoint_path = Path(config['checkpoint_path'])
        if not checkpoint_path.exists():
            logger.info(f"Downloading checkpoint to {checkpoint_path.parent}")
            snapshot_download(repo_id='cuhkaih/rhofold', local_dir=str(checkpoint_path.parent))

        # Load model weights
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['model'])
        model.eval()

        # Setup device
        device = get_device(config['device'])
        logger.info(f"Using device: {device}")
        model = model.to(device)

        # Prepare input data with MSA
        logger.info("Processing input sequence and MSA...")
        with timing('RNA 3D Structure Prediction with MSA', logger=logger):
            # Get features from FASTA and MSA files
            data_dict = get_features(str(fasta_file), str(msa_file))

            # Forward pass
            logger.info("Running model inference...")
            outputs = model(
                tokens=data_dict['tokens'].to(device),
                rna_fm_tokens=data_dict['rna_fm_tokens'].to(device),
                seq=data_dict['seq'],
            )

            output = outputs[-1]

        # Save results
        logger.info("Saving prediction results...")
        results = {}

        # Secondary structure (.ct format)
        ss_prob_map = torch.sigmoid(output['ss'][0, 0]).data.cpu().numpy()
        ss_file = output_dir / 'ss.ct'
        save_ss2ct(ss_prob_map, data_dict['seq'], str(ss_file), threshold=0.5)
        logger.info(f"Secondary structure saved: {ss_file}")
        results['secondary_structure'] = str(ss_file)

        # Distance maps and confidence (.npz format)
        npz_file = output_dir / 'results.npz'
        np.savez_compressed(
            npz_file,
            dist_n=torch.softmax(output['n'].squeeze(0), dim=0).data.cpu().numpy(),
            dist_p=torch.softmax(output['p'].squeeze(0), dim=0).data.cpu().numpy(),
            dist_c=torch.softmax(output['c4_'].squeeze(0), dim=0).data.cpu().numpy(),
            ss_prob_map=ss_prob_map,
            plddt=output['plddt'][0].data.cpu().numpy(),
        )
        logger.info(f"Distance maps and confidence scores saved: {npz_file}")
        results['distance_maps'] = str(npz_file)

        # 3D structure (.pdb format)
        unrelaxed_model = output_dir / 'unrelaxed_model.pdb'
        node_cords_pred = output['cord_tns_pred'][-1].squeeze(0)
        model.structure_module.converter.export_pdb_file(
            data_dict['seq'],
            node_cords_pred.data.cpu().numpy(),
            path=str(unrelaxed_model),
            chain_id=None,
            confidence=output['plddt'][0].data.cpu().numpy(),
            logger=logger
        )
        logger.info(f"Unrelaxed 3D structure saved: {unrelaxed_model}")
        results['unrelaxed_structure'] = str(unrelaxed_model)

        # Calculate average confidence score
        avg_plddt = float(output['plddt'][0].data.cpu().numpy().mean())
        logger.info(f"Average confidence (pLDDT): {avg_plddt:.2f}")

        # Amber relaxation
        if config['relax_steps'] > 0:
            with timing(f'Amber Relaxation: {config["relax_steps"]} iterations', logger=logger):
                amber_relax = AmberRelaxation(max_iterations=config['relax_steps'], logger=logger)
                relaxed_model = output_dir / f'relaxed_{config["relax_steps"]}_model.pdb'
                amber_relax.process(str(unrelaxed_model), str(relaxed_model))
                logger.info(f"Relaxed 3D structure saved: {relaxed_model}")
                results['relaxed_structure'] = str(relaxed_model)
        else:
            logger.info("Amber relaxation skipped (relax_steps = 0)")
            results['relaxed_structure'] = None

        logger.info("=" * 70)
        logger.info("RNA 3D structure prediction completed successfully!")
        logger.info(f"Results saved in: {output_dir}")
        logger.info(f"Average confidence score: {avg_plddt:.2f}")
        logger.info(f"MSA sequences used: {msa_count}")
        logger.info("=" * 70)

        return {
            "result": results,
            "output_dir": str(output_dir),
            "metadata": {
                "fasta_file": str(fasta_file),
                "msa_file": str(msa_file),
                "config": config,
                "msa_sequences": msa_count,
                "avg_confidence": avg_plddt
            }
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--fasta', '-f', required=True, help='Input FASTA file path')
    parser.add_argument('--msa', '-m', required=True, help='Input MSA file path (A3M format)')
    parser.add_argument('--output', '-o', help='Output directory path')
    parser.add_argument('--config', '-c', help='Config file (JSON)')
    parser.add_argument('--device', help='Device for computation (cpu, cuda:0, etc.)')
    parser.add_argument('--checkpoint', help='Path to pretrained model checkpoint')
    parser.add_argument('--relax-steps', type=int, help='Number of Amber relaxation steps')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override config with CLI arguments
    overrides = {}
    if args.device:
        overrides['device'] = args.device
    if args.checkpoint:
        overrides['checkpoint_path'] = args.checkpoint
    if args.relax_steps is not None:
        overrides['relax_steps'] = args.relax_steps

    # Run
    try:
        result = run_msa_prediction(
            fasta_file=args.fasta,
            msa_file=args.msa,
            output_file=args.output,
            config=config,
            **overrides
        )

        print(f"✅ Success: Results saved in {result['output_dir']}")
        print(f"Average confidence: {result['metadata']['avg_confidence']:.2f}")
        print(f"MSA sequences used: {result['metadata']['msa_sequences']}")
        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()