#!/usr/bin/env python3
"""
Script: rna_batch_prediction.py
Description: Clean RNA 3D structure prediction for multiple sequences in batch mode

Original Use Case: examples/use_case_3_batch_prediction.py
Dependencies Removed: Inlined all utility functions, simplified batch processing

Usage:
    python scripts/rna_batch_prediction.py --input_dir <directory> --output <output_dir>
    python scripts/rna_batch_prediction.py --multi_fasta <fasta_file> --output <output_dir>

Example:
    python scripts/rna_batch_prediction.py --input_dir examples/data --output results/batch_results --max_sequences 3
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import os
import sys
import logging
import json
import glob
import shutil
from pathlib import Path
from typing import Union, Optional, Dict, Any, List

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "device": "cpu",
    "checkpoint_path": "./pretrained/rhofold_pretrained_params.pt",
    "relax_steps": 1000,
    "max_sequences": None,
    "log_level": "INFO"
}

# ==============================================================================
# Inlined Utility Functions (simplified from repo)
# ==============================================================================
def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration. Inlined from original use case."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('RhoFold_Batch')
    logger.setLevel(getattr(logging, log_level.upper()))

    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

    # File handler
    file_handler = logging.FileHandler(output_dir / 'batch_log.txt', mode='w')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(getattr(logging, log_level.upper()))
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def find_sequence_pairs(input_dir: Path) -> List[Dict[str, Any]]:
    """Find FASTA and corresponding A3M files in input directory. Inlined from original."""
    fasta_files = list(input_dir.rglob("*.fasta"))
    pairs = []

    for fasta_file in fasta_files:
        base_name = fasta_file.stem
        dir_name = fasta_file.parent

        # Look for corresponding A3M file
        a3m_file = dir_name / f"{base_name}.a3m"

        if a3m_file.exists():
            pairs.append({
                'name': base_name,
                'fasta': str(fasta_file),
                'msa': str(a3m_file)
            })
        else:
            pairs.append({
                'name': base_name,
                'fasta': str(fasta_file),
                'msa': None  # Single sequence prediction
            })

    return pairs

def parse_multi_fasta(fasta_file: Path) -> List[Dict[str, str]]:
    """Parse multi-FASTA file into individual sequences. Inlined from original."""
    sequences = []
    current_header = None
    current_seq = ""

    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append({
                        'name': current_header[1:].split()[0],  # Remove '>' and take first part
                        'header': current_header,
                        'sequence': current_seq
                    })
                current_header = line
                current_seq = ""
            else:
                current_seq += line

        # Add the last sequence
        if current_header is not None:
            sequences.append({
                'name': current_header[1:].split()[0],
                'header': current_header,
                'sequence': current_seq
            })

    return sequences

def create_temp_fasta(sequence_data: Dict[str, str], temp_dir: Path) -> str:
    """Create temporary FASTA file for a single sequence. Inlined from original."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = temp_dir / f"{sequence_data['name']}.fasta"

    with open(fasta_path, 'w') as f:
        f.write(f"{sequence_data['header']}\n")
        f.write(f"{sequence_data['sequence']}\n")

    return str(fasta_path)

def load_rhofold_modules():
    """Lazy load RhoFold modules to minimize startup time."""
    try:
        # Add repo path to Python path for imports
        script_dir = Path(__file__).parent
        repo_path = script_dir.parent / "repo" / "RhoFold"
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
            "Activate with: mamba run -p ./env_py37 python scripts/rna_batch_prediction.py ..."
        )

def run_single_prediction(
    name: str,
    fasta_file: str,
    msa_file: Optional[str],
    output_dir: Path,
    model: Any,
    device: Any,
    config: Dict[str, Any],
    logger: logging.Logger,
    modules: Dict[str, Any]
) -> Dict[str, Any]:
    """Run prediction for a single sequence. Extracted from original batch function."""
    seq_output_dir = output_dir / name
    seq_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {name}...")

    try:
        # Extract modules for easier access
        get_features = modules['get_features']
        timing = modules['timing']
        save_ss2ct = modules['save_ss2ct']
        AmberRelaxation = modules['AmberRelaxation']
        torch = modules['torch']
        np = modules['np']

        # Prepare input data
        input_msa = msa_file if msa_file else fasta_file  # Use FASTA as MSA for single sequence
        data_dict = get_features(fasta_file, input_msa)

        # Forward pass
        with timing(f'Prediction for {name}', logger=logger):
            outputs = model(
                tokens=data_dict['tokens'].to(device),
                rna_fm_tokens=data_dict['rna_fm_tokens'].to(device),
                seq=data_dict['seq'],
            )

            output = outputs[-1]

        # Save results
        results = {}

        # Secondary structure
        ss_prob_map = torch.sigmoid(output['ss'][0, 0]).data.cpu().numpy()
        ss_file = seq_output_dir / 'ss.ct'
        save_ss2ct(ss_prob_map, data_dict['seq'], str(ss_file), threshold=0.5)
        results['secondary_structure'] = str(ss_file)

        # Distance maps and confidence
        npz_file = seq_output_dir / 'results.npz'
        np.savez_compressed(
            npz_file,
            dist_n=torch.softmax(output['n'].squeeze(0), dim=0).data.cpu().numpy(),
            dist_p=torch.softmax(output['p'].squeeze(0), dim=0).data.cpu().numpy(),
            dist_c=torch.softmax(output['c4_'].squeeze(0), dim=0).data.cpu().numpy(),
            ss_prob_map=ss_prob_map,
            plddt=output['plddt'][0].data.cpu().numpy(),
        )
        results['distance_maps'] = str(npz_file)

        # 3D structure
        unrelaxed_model = seq_output_dir / 'unrelaxed_model.pdb'
        node_cords_pred = output['cord_tns_pred'][-1].squeeze(0)
        model.structure_module.converter.export_pdb_file(
            data_dict['seq'],
            node_cords_pred.data.cpu().numpy(),
            path=str(unrelaxed_model),
            chain_id=None,
            confidence=output['plddt'][0].data.cpu().numpy(),
            logger=logger
        )
        results['unrelaxed_structure'] = str(unrelaxed_model)

        # Calculate confidence
        avg_plddt = float(output['plddt'][0].data.cpu().numpy().mean())

        # Amber relaxation
        if config['relax_steps'] > 0:
            amber_relax = AmberRelaxation(max_iterations=config['relax_steps'], logger=logger)
            relaxed_model = seq_output_dir / f'relaxed_{config["relax_steps"]}_model.pdb'
            amber_relax.process(str(unrelaxed_model), str(relaxed_model))
            results['relaxed_structure'] = str(relaxed_model)
        else:
            results['relaxed_structure'] = None

        logger.info(f"✓ {name} completed (confidence: {avg_plddt:.2f})")

        return {
            'name': name,
            'status': 'success',
            'confidence': avg_plddt,
            'output_dir': str(seq_output_dir),
            'results': results
        }

    except Exception as e:
        logger.error(f"✗ {name} failed: {e}")
        return {
            'name': name,
            'status': 'failed',
            'error': str(e),
            'output_dir': str(seq_output_dir)
        }

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_batch_prediction(
    input_dir: Optional[Union[str, Path]] = None,
    multi_fasta: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for batch RNA 3D structure prediction.

    Args:
        input_dir: Directory containing FASTA (and optionally A3M) files
        multi_fasta: Multi-FASTA file containing multiple sequences
        output_file: Path to save output directory
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        include: List of sequence names to include (default: all)
        exclude: List of sequence names to exclude
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - results: List of individual prediction results
            - output_dir: Path to output directory
            - metadata: Execution metadata

    Example:
        >>> result = run_batch_prediction(input_dir="examples/data", output_file="batch_results")
        >>> print(result['output_dir'])
    """
    # Setup
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    if not input_dir and not multi_fasta:
        raise ValueError("Must specify either input_dir or multi_fasta")

    # Setup output directory
    if output_file:
        output_dir = Path(output_file)
    else:
        output_dir = Path("output") / "batch_prediction"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir, config.get('log_level', 'INFO'))

    logger.info("=" * 80)
    logger.info("RhoFold+ Batch RNA 3D Structure Prediction")
    logger.info("=" * 80)

    # Find sequences to process
    sequences = []

    if input_dir:
        input_dir = Path(input_dir)
        logger.info(f"Scanning directory: {input_dir}")
        pairs = find_sequence_pairs(input_dir)
        sequences = [{'name': p['name'], 'fasta': p['fasta'], 'msa': p['msa']} for p in pairs]
        logger.info(f"Found {len(sequences)} sequence files")

    elif multi_fasta:
        multi_fasta = Path(multi_fasta)
        logger.info(f"Parsing multi-FASTA: {multi_fasta}")
        fasta_sequences = parse_multi_fasta(multi_fasta)
        temp_dir = output_dir / 'temp_fasta'

        for seq_data in fasta_sequences:
            temp_fasta = create_temp_fasta(seq_data, temp_dir)
            sequences.append({
                'name': seq_data['name'],
                'fasta': temp_fasta,
                'msa': None  # Single sequence prediction
            })
        logger.info(f"Found {len(sequences)} sequences in multi-FASTA")

    # Filter sequences
    if include:
        sequences = [s for s in sequences if s['name'] in include]
        logger.info(f"Including only: {include}")

    if exclude:
        sequences = [s for s in sequences if s['name'] not in exclude]
        logger.info(f"Excluding: {exclude}")

    if config['max_sequences']:
        sequences = sequences[:config['max_sequences']]
        logger.info(f"Limited to first {config['max_sequences']} sequences")

    if not sequences:
        raise ValueError("No sequences found to process!")

    logger.info(f"Processing {len(sequences)} sequences")

    # Load RhoFold modules
    modules = load_rhofold_modules()

    # Extract modules for easier access
    RhoFold = modules['RhoFold']
    rhofold_config = modules['rhofold_config']
    get_device = modules['get_device']
    snapshot_download = modules['snapshot_download']
    torch = modules['torch']

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

        # Process sequences
        results = []
        successful = 0
        failed = 0

        for i, seq_info in enumerate(sequences, 1):
            logger.info(f"Processing sequence {i}/{len(sequences)}: {seq_info['name']}")

            result = run_single_prediction(
                name=seq_info['name'],
                fasta_file=seq_info['fasta'],
                msa_file=seq_info['msa'],
                output_dir=output_dir,
                model=model,
                device=device,
                config=config,
                logger=logger,
                modules=modules
            )

            results.append(result)

            if result['status'] == 'success':
                successful += 1
            else:
                failed += 1

        # Write batch summary
        summary_file = output_dir / 'batch_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("RhoFold+ Batch Prediction Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total sequences: {len(sequences)}\n")
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {failed}\n")
            f.write(f"Success rate: {successful/len(sequences)*100:.1f}%\n\n")

            f.write("Individual Results:\n")
            f.write("-" * 20 + "\n")
            for result in results:
                status_symbol = "✓" if result['status'] == 'success' else "✗"
                if result['status'] == 'success':
                    f.write(f"{status_symbol} {result['name']}: confidence {result['confidence']:.2f}\n")
                else:
                    f.write(f"{status_symbol} {result['name']}: {result['error']}\n")

        logger.info("=" * 80)
        logger.info("Batch prediction completed!")
        logger.info(f"Results saved in: {output_dir}")
        logger.info(f"Successful: {successful}/{len(sequences)} ({successful/len(sequences)*100:.1f}%)")
        logger.info("=" * 80)

        # Clean up temp files if created
        temp_dir = output_dir / 'temp_fasta'
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary files")

        return {
            "results": results,
            "output_dir": str(output_dir),
            "metadata": {
                "total_sequences": len(sequences),
                "successful": successful,
                "failed": failed,
                "success_rate": successful/len(sequences)*100,
                "config": config
            }
        }

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
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

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_dir', help='Directory containing FASTA (and optionally A3M) files')
    input_group.add_argument('--multi_fasta', help='Multi-FASTA file containing multiple sequences')

    # Output and filtering
    parser.add_argument('--output', '-o', required=True, help='Output directory path')
    parser.add_argument('--include', help='Comma-separated list of sequence names to process')
    parser.add_argument('--exclude', help='Comma-separated list of sequence names to skip')
    parser.add_argument('--max_sequences', type=int, help='Maximum number of sequences to process')

    # Configuration
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
    if args.max_sequences is not None:
        overrides['max_sequences'] = args.max_sequences

    # Parse include/exclude lists
    include = None
    exclude = None
    if args.include:
        include = [name.strip() for name in args.include.split(',')]
    if args.exclude:
        exclude = [name.strip() for name in args.exclude.split(',')]

    # Run
    try:
        result = run_batch_prediction(
            input_dir=args.input_dir,
            multi_fasta=args.multi_fasta,
            output_file=args.output,
            config=config,
            include=include,
            exclude=exclude,
            **overrides
        )

        print(f"✅ Success: Processed {result['metadata']['total_sequences']} sequences")
        print(f"Success rate: {result['metadata']['success_rate']:.1f}%")
        print(f"Results saved in {result['output_dir']}")
        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()