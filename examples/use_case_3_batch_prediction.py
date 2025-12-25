#!/usr/bin/env python3
"""
RhoFold+ Use Case 3: Batch RNA 3D Structure Prediction

This script performs RNA 3D structure prediction for multiple sequences in batch mode.
It can process either multiple FASTA files or a single file with multiple sequences.

Input: Directory with FASTA/A3M pairs OR multi-FASTA file
Output: Organized results for each sequence
"""

import os
import sys
import argparse
import logging
import glob
from pathlib import Path

def setup_logging(output_dir):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger('RhoFold_Batch')
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

    # File handler
    file_handler = logging.FileHandler(f'{output_dir}/batch_log.txt', mode='w')
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def find_sequence_pairs(input_dir):
    """Find FASTA and corresponding A3M files in input directory"""
    fasta_files = glob.glob(os.path.join(input_dir, "**", "*.fasta"), recursive=True)
    pairs = []

    for fasta_file in fasta_files:
        base_name = Path(fasta_file).stem
        dir_name = Path(fasta_file).parent

        # Look for corresponding A3M file
        a3m_file = os.path.join(dir_name, f"{base_name}.a3m")

        if os.path.exists(a3m_file):
            pairs.append({
                'name': base_name,
                'fasta': fasta_file,
                'msa': a3m_file
            })
        else:
            pairs.append({
                'name': base_name,
                'fasta': fasta_file,
                'msa': None  # Single sequence prediction
            })

    return pairs

def parse_multi_fasta(fasta_file):
    """Parse multi-FASTA file into individual sequences"""
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

def create_temp_fasta(sequence_data, temp_dir):
    """Create temporary FASTA file for a single sequence"""
    os.makedirs(temp_dir, exist_ok=True)
    fasta_path = os.path.join(temp_dir, f"{sequence_data['name']}.fasta")

    with open(fasta_path, 'w') as f:
        f.write(f"{sequence_data['header']}\n")
        f.write(f"{sequence_data['sequence']}\n")

    return fasta_path

def run_single_prediction(name, fasta_file, msa_file, output_dir, device, ckpt, relax_steps, logger, model):
    """Run prediction for a single sequence"""
    seq_output_dir = os.path.join(output_dir, name)
    os.makedirs(seq_output_dir, exist_ok=True)

    logger.info(f"Processing {name}...")

    try:
        # Import required functions
        from rhofold.utils import get_device, timing, save_ss2ct
        from rhofold.utils.alphabet import get_features
        from rhofold.relax.relax import AmberRelaxation
        import torch
        import numpy as np

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
        # Secondary structure
        ss_prob_map = torch.sigmoid(output['ss'][0, 0]).data.cpu().numpy()
        ss_file = f'{seq_output_dir}/ss.ct'
        save_ss2ct(ss_prob_map, data_dict['seq'], ss_file, threshold=0.5)

        # Distance maps and confidence
        npz_file = f'{seq_output_dir}/results.npz'
        np.savez_compressed(
            npz_file,
            dist_n=torch.softmax(output['n'].squeeze(0), dim=0).data.cpu().numpy(),
            dist_p=torch.softmax(output['p'].squeeze(0), dim=0).data.cpu().numpy(),
            dist_c=torch.softmax(output['c4_'].squeeze(0), dim=0).data.cpu().numpy(),
            ss_prob_map=ss_prob_map,
            plddt=output['plddt'][0].data.cpu().numpy(),
        )

        # 3D structure
        unrelaxed_model = f'{seq_output_dir}/unrelaxed_model.pdb'
        node_cords_pred = output['cord_tns_pred'][-1].squeeze(0)
        model.structure_module.converter.export_pdb_file(
            data_dict['seq'],
            node_cords_pred.data.cpu().numpy(),
            path=unrelaxed_model,
            chain_id=None,
            confidence=output['plddt'][0].data.cpu().numpy(),
            logger=logger
        )

        # Calculate confidence
        avg_plddt = float(output['plddt'][0].data.cpu().numpy().mean())

        # Amber relaxation
        relaxed_model = None
        if relax_steps > 0:
            amber_relax = AmberRelaxation(max_iterations=relax_steps, logger=logger)
            relaxed_model = f'{seq_output_dir}/relaxed_{relax_steps}_model.pdb'
            amber_relax.process(unrelaxed_model, relaxed_model)

        logger.info(f"✓ {name} completed (confidence: {avg_plddt:.2f})")

        return {
            'name': name,
            'status': 'success',
            'confidence': avg_plddt,
            'output_dir': seq_output_dir,
            'unrelaxed_pdb': unrelaxed_model,
            'relaxed_pdb': relaxed_model,
            'secondary_structure': ss_file,
            'results_npz': npz_file
        }

    except Exception as e:
        logger.error(f"✗ {name} failed: {e}")
        return {
            'name': name,
            'status': 'failed',
            'error': str(e),
            'output_dir': seq_output_dir
        }

def main():
    parser = argparse.ArgumentParser(
        description="RhoFold+ Batch RNA 3D Structure Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Batch predict all examples in data directory
    python use_case_3_batch_prediction.py --input_dir examples/data --output output/batch_results

    # Process multi-FASTA file
    python use_case_3_batch_prediction.py --multi_fasta my_sequences.fasta --output output/multi_fasta_results

    # Use GPU and skip relaxation for faster processing
    python use_case_3_batch_prediction.py --input_dir examples/data --output output/batch_fast --device cuda:0 --relax_steps 0

    # Process only specific sequences
    python use_case_3_batch_prediction.py --input_dir examples/data --output output/subset --include "3owzA,5ddoA"

Output structure:
    output/
    ├── batch_log.txt           # Overall batch log
    ├── batch_summary.txt       # Summary of all predictions
    ├── sequence1/              # Results for sequence 1
    │   ├── unrelaxed_model.pdb
    │   ├── relaxed_1000_model.pdb
    │   ├── ss.ct
    │   └── results.npz
    └── sequence2/              # Results for sequence 2
        └── ...
        """
    )

    parser.add_argument(
        "--input_dir",
        help="Directory containing FASTA (and optionally A3M) files",
        type=str
    )

    parser.add_argument(
        "--multi_fasta",
        help="Multi-FASTA file containing multiple sequences",
        type=str
    )

    parser.add_argument(
        "--output", "-o",
        help="Output directory for batch results",
        required=True,
        type=str
    )

    parser.add_argument(
        "--include",
        help="Comma-separated list of sequence names to process (default: all)",
        type=str
    )

    parser.add_argument(
        "--exclude",
        help="Comma-separated list of sequence names to skip",
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
        "--max_sequences",
        help="Maximum number of sequences to process",
        default=None,
        type=int
    )

    args = parser.parse_args()

    # Validate input arguments
    if not args.input_dir and not args.multi_fasta:
        print("Error: Must specify either --input_dir or --multi_fasta")
        sys.exit(1)

    # Setup logging
    logger = setup_logging(args.output)

    logger.info("=" * 80)
    logger.info("RhoFold+ Batch RNA 3D Structure Prediction")
    logger.info("=" * 80)

    # Find sequences to process
    sequences = []

    if args.input_dir:
        logger.info(f"Scanning directory: {args.input_dir}")
        pairs = find_sequence_pairs(args.input_dir)
        sequences = [{'name': p['name'], 'fasta': p['fasta'], 'msa': p['msa']} for p in pairs]
        logger.info(f"Found {len(sequences)} sequence files")

    elif args.multi_fasta:
        logger.info(f"Parsing multi-FASTA: {args.multi_fasta}")
        fasta_sequences = parse_multi_fasta(args.multi_fasta)
        temp_dir = os.path.join(args.output, 'temp_fasta')

        for seq_data in fasta_sequences:
            temp_fasta = create_temp_fasta(seq_data, temp_dir)
            sequences.append({
                'name': seq_data['name'],
                'fasta': temp_fasta,
                'msa': None  # Single sequence prediction
            })
        logger.info(f"Found {len(sequences)} sequences in multi-FASTA")

    # Filter sequences
    if args.include:
        include_list = [name.strip() for name in args.include.split(',')]
        sequences = [s for s in sequences if s['name'] in include_list]
        logger.info(f"Including only: {include_list}")

    if args.exclude:
        exclude_list = [name.strip() for name in args.exclude.split(',')]
        sequences = [s for s in sequences if s['name'] not in exclude_list]
        logger.info(f"Excluding: {exclude_list}")

    if args.max_sequences:
        sequences = sequences[:args.max_sequences]
        logger.info(f"Limited to first {args.max_sequences} sequences")

    if not sequences:
        logger.error("No sequences found to process!")
        sys.exit(1)

    logger.info(f"Processing {len(sequences)} sequences")

    # Import RhoFold modules
    try:
        # Add repo path to Python path for imports
        repo_path = os.path.join(os.path.dirname(__file__), '..', 'repo', 'RhoFold')
        sys.path.insert(0, repo_path)

        from rhofold.rhofold import RhoFold
        from rhofold.config import rhofold_config
        from rhofold.utils import get_device
        import torch
        from huggingface_hub import snapshot_download

    except ImportError as e:
        logger.error(f"Failed to import RhoFold modules: {e}")
        logger.error("Make sure you're running this from the RhoFold conda environment (env_py37)")
        sys.exit(1)

    try:
        # Load model once for all predictions
        logger.info("Loading RhoFold model...")
        model = RhoFold(rhofold_config)

        # Download checkpoint if needed
        if not os.path.exists(args.ckpt):
            logger.info(f"Downloading checkpoint to {Path(args.ckpt).parent}")
            snapshot_download(repo_id='cuhkaih/rhofold', local_dir=Path(args.ckpt).parent)

        # Load model weights
        model.load_state_dict(torch.load(args.ckpt, map_location=torch.device('cpu'))['model'])
        model.eval()

        # Setup device
        device = get_device(args.device)
        logger.info(f"Using device: {device}")
        model = model.to(device)

        # Process sequences
        results = []
        failed_count = 0
        success_count = 0

        for i, seq_info in enumerate(sequences, 1):
            logger.info(f"\n--- Processing {i}/{len(sequences)}: {seq_info['name']} ---")

            result = run_single_prediction(
                seq_info['name'],
                seq_info['fasta'],
                seq_info['msa'],
                args.output,
                device,
                args.ckpt,
                args.relax_steps,
                logger,
                model
            )

            results.append(result)

            if result['status'] == 'success':
                success_count += 1
            else:
                failed_count += 1

        # Generate summary
        logger.info("\n" + "=" * 80)
        logger.info("BATCH PREDICTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total sequences processed: {len(sequences)}")
        logger.info(f"Successful predictions: {success_count}")
        logger.info(f"Failed predictions: {failed_count}")

        # Write detailed summary to file
        summary_file = os.path.join(args.output, 'batch_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("RhoFold+ Batch Prediction Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total sequences: {len(sequences)}\n")
            f.write(f"Successful: {success_count}\n")
            f.write(f"Failed: {failed_count}\n")
            f.write(f"Success rate: {100*success_count/len(sequences):.1f}%\n\n")

            f.write("Detailed Results:\n")
            f.write("-" * 50 + "\n")

            for result in results:
                f.write(f"\nSequence: {result['name']}\n")
                f.write(f"Status: {result['status']}\n")

                if result['status'] == 'success':
                    f.write(f"Confidence: {result['confidence']:.2f}\n")
                    f.write(f"Output directory: {result['output_dir']}\n")
                    f.write(f"PDB files: {result['unrelaxed_pdb']}")
                    if result['relaxed_pdb']:
                        f.write(f", {result['relaxed_pdb']}")
                    f.write("\n")
                else:
                    f.write(f"Error: {result['error']}\n")

        logger.info(f"Detailed summary saved: {summary_file}")

        if success_count > 0:
            avg_confidence = sum(r['confidence'] for r in results if r['status'] == 'success') / success_count
            logger.info(f"Average confidence: {avg_confidence:.2f}")

        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()