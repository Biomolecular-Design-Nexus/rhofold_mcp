#!/usr/bin/env python3
"""
RhoFold+ Use Case 1: Single Sequence RNA 3D Structure Prediction

This script performs RNA 3D structure prediction using only the input sequence,
without MSA (Multiple Sequence Alignment). This is the fastest but least accurate mode.

Input: RNA sequence in FASTA format
Output: 3D structure (PDB), secondary structure (CT), distance maps (NPZ)
"""

import os
import sys
import argparse
import logging
from pathlib import Path

def setup_logging(output_dir):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger('RhoFold_SingleSeq')
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

    # File handler
    file_handler = logging.FileHandler(f'{output_dir}/log.txt', mode='w')
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def validate_rna_sequence(fasta_file):
    """Validate RNA sequence contains only valid nucleotides (A, U, G, C)"""
    valid_nucleotides = set(['A', 'U', 'G', 'C', 'N'])  # N for unknown

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

def main():
    parser = argparse.ArgumentParser(
        description="RhoFold+ Single Sequence RNA 3D Structure Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Predict structure for 3owzA example
    python use_case_1_single_sequence_prediction.py --input examples/data/3owzA/3owzA.fasta --output output/3owzA_single

    # Use GPU for faster prediction
    python use_case_1_single_sequence_prediction.py --input examples/data/5ddoA/5ddoA.fasta --output output/5ddoA_single --device cuda:0

Output files:
    unrelaxed_model.pdb     - Predicted 3D structure (unrelaxed)
    relaxed_1000_model.pdb  - Amber-relaxed 3D structure
    ss.ct                   - Predicted secondary structure
    results.npz             - Distance maps and confidence scores
    log.txt                 - Execution log
        """
    )

    parser.add_argument(
        "--input", "-i",
        help="Path to input RNA sequence in FASTA format",
        default="examples/data/3owzA/3owzA.fasta",
        type=str
    )

    parser.add_argument(
        "--output", "-o",
        help="Output directory for results",
        default="output/single_sequence_prediction",
        type=str
    )

    parser.add_argument(
        "--device",
        help="Device for computation (cpu, cuda:0, etc.)",
        default="cpu",
        type=str
    )

    parser.add_argument(
        "--ckpt",
        help="Path to pretrained model checkpoint",
        default="./pretrained/rhofold_pretrained_params.pt",
        type=str
    )

    parser.add_argument(
        "--relax_steps",
        help="Number of steps for Amber relaxation (0 to skip)",
        default=1000,
        type=int
    )

    parser.add_argument(
        "--validate",
        help="Validate RNA sequence before prediction",
        action="store_true",
        default=True
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        sys.exit(1)

    # Setup logging
    logger = setup_logging(args.output)

    logger.info("=" * 60)
    logger.info("RhoFold+ Single Sequence RNA 3D Structure Prediction")
    logger.info("=" * 60)
    logger.info(f"Input FASTA: {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Checkpoint: {args.ckpt}")
    logger.info(f"Amber relaxation steps: {args.relax_steps}")

    # Validate RNA sequence
    if args.validate:
        try:
            sequence = validate_rna_sequence(args.input)
            logger.info(f"RNA sequence validated: {len(sequence)} nucleotides")
            logger.info(f"Sequence preview: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
        except Exception as e:
            logger.error(f"Sequence validation failed: {e}")
            sys.exit(1)

    # Import RhoFold modules (done here to allow script to run even if RhoFold not installed)
    try:
        # Add repo path to Python path for imports
        repo_path = os.path.join(os.path.dirname(__file__), '..', 'repo', 'RhoFold')
        sys.path.insert(0, repo_path)

        from rhofold.rhofold import RhoFold
        from rhofold.config import rhofold_config
        from rhofold.utils import get_device, timing
        from rhofold.utils.alphabet import get_features
        from rhofold.relax.relax import AmberRelaxation

        import torch
        import numpy as np
        from huggingface_hub import snapshot_download

    except ImportError as e:
        logger.error(f"Failed to import RhoFold modules: {e}")
        logger.error("Make sure you're running this from the RhoFold conda environment (env_py37)")
        logger.error("Activate with: mamba run -p ./env_py37 python examples/use_case_1_single_sequence_prediction.py ...")
        sys.exit(1)

    try:
        # Load model
        logger.info("Loading RhoFold model...")
        model = RhoFold(rhofold_config)

        # Download checkpoint if needed
        if not os.path.exists(args.ckpt):
            logger.info(f"Downloading checkpoint to {Path(args.ckpt).parent}")
            snapshot_download(repo_id='cuhkaih/rhofold', local_dir=Path(args.ckpt).parent)

        # Load model weights
        logger.info(f"Loading checkpoint: {args.ckpt}")
        model.load_state_dict(torch.load(args.ckpt, map_location=torch.device('cpu'))['model'])
        model.eval()

        # Setup device
        device = get_device(args.device)
        logger.info(f"Using device: {device}")
        model = model.to(device)

        # Prepare input data for single sequence prediction
        with timing('Single Sequence Prediction', logger=logger):
            # For single sequence prediction, we use the FASTA file as both sequence and MSA
            data_dict = get_features(args.input, args.input)  # Same file for both

            # Forward pass
            outputs = model(
                tokens=data_dict['tokens'].to(device),
                rna_fm_tokens=data_dict['rna_fm_tokens'].to(device),
                seq=data_dict['seq'],
            )

            output = outputs[-1]

        # Save results
        logger.info("Saving prediction results...")

        # Secondary structure (.ct format)
        from rhofold.utils import save_ss2ct
        ss_prob_map = torch.sigmoid(output['ss'][0, 0]).data.cpu().numpy()
        ss_file = f'{args.output}/ss.ct'
        save_ss2ct(ss_prob_map, data_dict['seq'], ss_file, threshold=0.5)
        logger.info(f"Secondary structure saved: {ss_file}")

        # Distance maps and confidence (.npz format)
        npz_file = f'{args.output}/results.npz'
        np.savez_compressed(
            npz_file,
            dist_n=torch.softmax(output['n'].squeeze(0), dim=0).data.cpu().numpy(),
            dist_p=torch.softmax(output['p'].squeeze(0), dim=0).data.cpu().numpy(),
            dist_c=torch.softmax(output['c4_'].squeeze(0), dim=0).data.cpu().numpy(),
            ss_prob_map=ss_prob_map,
            plddt=output['plddt'][0].data.cpu().numpy(),
        )
        logger.info(f"Distance maps and confidence scores saved: {npz_file}")

        # 3D structure (.pdb format)
        unrelaxed_model = f'{args.output}/unrelaxed_model.pdb'
        node_cords_pred = output['cord_tns_pred'][-1].squeeze(0)
        model.structure_module.converter.export_pdb_file(
            data_dict['seq'],
            node_cords_pred.data.cpu().numpy(),
            path=unrelaxed_model,
            chain_id=None,
            confidence=output['plddt'][0].data.cpu().numpy(),
            logger=logger
        )
        logger.info(f"Unrelaxed 3D structure saved: {unrelaxed_model}")

        # Amber relaxation
        if args.relax_steps > 0:
            with timing(f'Amber Relaxation: {args.relax_steps} iterations', logger=logger):
                amber_relax = AmberRelaxation(max_iterations=args.relax_steps, logger=logger)
                relaxed_model = f'{args.output}/relaxed_{args.relax_steps}_model.pdb'
                amber_relax.process(unrelaxed_model, relaxed_model)
                logger.info(f"Relaxed 3D structure saved: {relaxed_model}")
        else:
            logger.info("Amber relaxation skipped (relax_steps = 0)")

        logger.info("=" * 60)
        logger.info("Single sequence prediction completed successfully!")
        logger.info(f"Results saved in: {args.output}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()