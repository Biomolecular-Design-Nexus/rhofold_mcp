#!/usr/bin/env python3
"""
RhoFold+ Use Case 4: RNA Structure Analysis and Validation

This script analyzes predicted RNA 3D structures and provides validation metrics.
It can compare multiple predictions, calculate quality metrics, and generate reports.

Input: RhoFold output directory(ies) or PDB files
Output: Analysis report with quality metrics and comparisons
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import glob

def setup_logging(output_dir):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger('RhoFold_Analysis')
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

    # File handler
    file_handler = logging.FileHandler(f'{output_dir}/analysis_log.txt', mode='w')
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def parse_pdb_file(pdb_file):
    """Parse PDB file and extract basic information"""
    atom_records = []
    sequence = ""
    confidence_scores = []

    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    # Extract atom information
                    atom_type = line[12:16].strip()
                    residue = line[17:20].strip()
                    chain = line[21:22].strip()
                    res_num = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    b_factor = float(line[60:66].strip())  # Confidence score in RhoFold

                    atom_records.append({
                        'atom_type': atom_type,
                        'residue': residue,
                        'chain': chain,
                        'res_num': res_num,
                        'x': x, 'y': y, 'z': z,
                        'confidence': b_factor
                    })

                    # Build sequence (use C4' atoms to avoid duplicates)
                    if atom_type == "C4'":
                        # Map RNA residues to single letters
                        rna_map = {'A': 'A', 'U': 'U', 'G': 'G', 'C': 'C'}
                        if residue in rna_map:
                            sequence += rna_map[residue]
                        confidence_scores.append(b_factor)

        return {
            'atoms': atom_records,
            'sequence': sequence,
            'confidence_scores': confidence_scores,
            'length': len(sequence)
        }

    except Exception as e:
        raise Exception(f"Failed to parse PDB file {pdb_file}: {e}")

def analyze_structure_quality(pdb_data):
    """Analyze structure quality metrics"""
    if not pdb_data['confidence_scores']:
        return {}

    confidence_scores = np.array(pdb_data['confidence_scores'])

    # Basic statistics
    quality_metrics = {
        'length': pdb_data['length'],
        'avg_confidence': float(np.mean(confidence_scores)),
        'min_confidence': float(np.min(confidence_scores)),
        'max_confidence': float(np.max(confidence_scores)),
        'std_confidence': float(np.std(confidence_scores)),
        'median_confidence': float(np.median(confidence_scores))
    }

    # Quality categories (based on AlphaFold confidence ranges)
    very_high = np.sum(confidence_scores >= 90)
    confident = np.sum((confidence_scores >= 70) & (confidence_scores < 90))
    low = np.sum((confidence_scores >= 50) & (confidence_scores < 70))
    very_low = np.sum(confidence_scores < 50)

    quality_metrics.update({
        'very_high_conf_residues': int(very_high),
        'confident_residues': int(confident),
        'low_conf_residues': int(low),
        'very_low_conf_residues': int(very_low),
        'very_high_conf_percent': float(100 * very_high / len(confidence_scores)),
        'confident_percent': float(100 * confident / len(confidence_scores)),
        'low_conf_percent': float(100 * low / len(confidence_scores)),
        'very_low_conf_percent': float(100 * very_low / len(confidence_scores))
    })

    return quality_metrics

def load_results_npz(npz_file):
    """Load and analyze RhoFold results.npz file"""
    try:
        data = np.load(npz_file)

        analysis = {
            'has_distance_maps': 'dist_n' in data and 'dist_p' in data and 'dist_c' in data,
            'has_secondary_structure': 'ss_prob_map' in data,
            'has_confidence': 'plddt' in data
        }

        if 'plddt' in data:
            plddt = data['plddt']
            analysis['plddt_stats'] = {
                'mean': float(np.mean(plddt)),
                'std': float(np.std(plddt)),
                'min': float(np.min(plddt)),
                'max': float(np.max(plddt))
            }

        if 'ss_prob_map' in data:
            ss_map = data['ss_prob_map']
            # Count predicted base pairs (threshold = 0.5)
            base_pairs = np.sum(ss_map > 0.5)
            analysis['predicted_base_pairs'] = int(base_pairs)
            analysis['ss_map_shape'] = ss_map.shape

        return analysis

    except Exception as e:
        raise Exception(f"Failed to load NPZ file {npz_file}: {e}")

def parse_secondary_structure_ct(ct_file):
    """Parse secondary structure from .ct file"""
    try:
        pairs = []
        sequence = ""

        with open(ct_file, 'r') as f:
            lines = f.readlines()

        # First line contains sequence length and name
        if not lines:
            return {'pairs': [], 'sequence': '', 'num_pairs': 0}

        # Skip header and parse structure
        for line in lines[1:]:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 6:
                    pos = int(parts[0])
                    nucleotide = parts[1]
                    pair_pos = int(parts[4])

                    sequence += nucleotide

                    if pair_pos > 0 and pair_pos > pos:  # Avoid duplicate pairs
                        pairs.append((pos, pair_pos))

        return {
            'pairs': pairs,
            'sequence': sequence,
            'num_pairs': len(pairs),
            'length': len(sequence)
        }

    except Exception as e:
        raise Exception(f"Failed to parse CT file {ct_file}: {e}")

def find_prediction_outputs(base_dir):
    """Find RhoFold output files in directory"""
    outputs = []

    # Look for individual prediction directories
    prediction_dirs = []

    if os.path.isdir(base_dir):
        # Check if this is a single prediction directory
        if os.path.exists(os.path.join(base_dir, 'unrelaxed_model.pdb')):
            prediction_dirs.append(base_dir)
        else:
            # Look for subdirectories with predictions
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'unrelaxed_model.pdb')):
                    prediction_dirs.append(item_path)

    for pred_dir in prediction_dirs:
        name = os.path.basename(pred_dir)
        output_info = {'name': name, 'directory': pred_dir}

        # Find output files
        unrelaxed_pdb = os.path.join(pred_dir, 'unrelaxed_model.pdb')
        relaxed_pdbs = glob.glob(os.path.join(pred_dir, 'relaxed_*_model.pdb'))
        results_npz = os.path.join(pred_dir, 'results.npz')
        ss_ct = os.path.join(pred_dir, 'ss.ct')

        output_info['files'] = {
            'unrelaxed_pdb': unrelaxed_pdb if os.path.exists(unrelaxed_pdb) else None,
            'relaxed_pdb': relaxed_pdbs[0] if relaxed_pdbs else None,
            'results_npz': results_npz if os.path.exists(results_npz) else None,
            'secondary_structure_ct': ss_ct if os.path.exists(ss_ct) else None
        }

        outputs.append(output_info)

    return outputs

def analyze_single_prediction(output_info, logger):
    """Analyze a single prediction output"""
    name = output_info['name']
    files = output_info['files']

    logger.info(f"Analyzing {name}...")

    analysis = {
        'name': name,
        'directory': output_info['directory'],
        'files_found': {},
        'structure_quality': {},
        'secondary_structure': {},
        'results_data': {}
    }

    # Check which files are available
    for file_type, file_path in files.items():
        analysis['files_found'][file_type] = file_path is not None
        if file_path and os.path.exists(file_path):
            analysis['files_found'][f"{file_type}_size"] = os.path.getsize(file_path)

    # Analyze PDB structure
    pdb_file = files['relaxed_pdb'] if files['relaxed_pdb'] else files['unrelaxed_pdb']
    if pdb_file and os.path.exists(pdb_file):
        try:
            pdb_data = parse_pdb_file(pdb_file)
            analysis['structure_quality'] = analyze_structure_quality(pdb_data)
            analysis['sequence'] = pdb_data['sequence']
            logger.info(f"  Structure: {pdb_data['length']} residues, avg confidence: {analysis['structure_quality'].get('avg_confidence', 'N/A'):.2f}")
        except Exception as e:
            logger.error(f"  Failed to analyze PDB: {e}")

    # Analyze secondary structure
    if files['secondary_structure_ct'] and os.path.exists(files['secondary_structure_ct']):
        try:
            ss_data = parse_secondary_structure_ct(files['secondary_structure_ct'])
            analysis['secondary_structure'] = ss_data
            logger.info(f"  Secondary structure: {ss_data['num_pairs']} base pairs")
        except Exception as e:
            logger.error(f"  Failed to analyze secondary structure: {e}")

    # Analyze results NPZ
    if files['results_npz'] and os.path.exists(files['results_npz']):
        try:
            npz_data = load_results_npz(files['results_npz'])
            analysis['results_data'] = npz_data
            logger.info(f"  Results data loaded successfully")
        except Exception as e:
            logger.error(f"  Failed to analyze NPZ data: {e}")

    return analysis

def generate_report(analyses, output_file):
    """Generate comprehensive analysis report"""
    with open(output_file, 'w') as f:
        f.write("RhoFold+ Structure Analysis Report\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total predictions analyzed: {len(analyses)}\n\n")

        # Summary table
        f.write("Summary Table:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Name':<20} {'Length':<8} {'Avg Conf':<10} {'Base Pairs':<12} {'Files':<20}\n")
        f.write("-" * 80 + "\n")

        for analysis in analyses:
            name = analysis['name'][:19]
            length = analysis['structure_quality'].get('length', 'N/A')
            avg_conf = analysis['structure_quality'].get('avg_confidence', 0)
            base_pairs = analysis['secondary_structure'].get('num_pairs', 'N/A')

            files_present = []
            if analysis['files_found'].get('unrelaxed_pdb'):
                files_present.append('unrel')
            if analysis['files_found'].get('relaxed_pdb'):
                files_present.append('rel')
            if analysis['files_found'].get('results_npz'):
                files_present.append('npz')
            if analysis['files_found'].get('secondary_structure_ct'):
                files_present.append('ss')

            files_str = ','.join(files_present)
            f.write(f"{name:<20} {length:<8} {avg_conf:<10.2f} {base_pairs:<12} {files_str:<20}\n")

        f.write("\n")

        # Detailed analysis for each prediction
        for analysis in analyses:
            f.write(f"\nDetailed Analysis: {analysis['name']}\n")
            f.write("-" * 40 + "\n")

            if analysis['structure_quality']:
                sq = analysis['structure_quality']
                f.write(f"Structure Quality:\n")
                f.write(f"  Length: {sq['length']} residues\n")
                f.write(f"  Average confidence: {sq['avg_confidence']:.2f}\n")
                f.write(f"  Confidence range: {sq['min_confidence']:.2f} - {sq['max_confidence']:.2f}\n")
                f.write(f"  Standard deviation: {sq['std_confidence']:.2f}\n")
                f.write(f"  Confidence distribution:\n")
                f.write(f"    Very high (≥90): {sq['very_high_conf_residues']} ({sq['very_high_conf_percent']:.1f}%)\n")
                f.write(f"    Confident (70-89): {sq['confident_residues']} ({sq['confident_percent']:.1f}%)\n")
                f.write(f"    Low (50-69): {sq['low_conf_residues']} ({sq['low_conf_percent']:.1f}%)\n")
                f.write(f"    Very low (<50): {sq['very_low_conf_residues']} ({sq['very_low_conf_percent']:.1f}%)\n")

            if analysis['secondary_structure']:
                ss = analysis['secondary_structure']
                f.write(f"\nSecondary Structure:\n")
                f.write(f"  Base pairs: {ss['num_pairs']}\n")
                f.write(f"  Sequence length: {ss['length']}\n")
                if ss['length'] > 0:
                    pairing_density = ss['num_pairs'] / ss['length']
                    f.write(f"  Pairing density: {pairing_density:.3f} pairs/nucleotide\n")

            if 'sequence' in analysis and analysis['sequence']:
                f.write(f"\nSequence: {analysis['sequence']}\n")

        # Overall statistics
        if analyses:
            f.write("\n" + "=" * 50 + "\n")
            f.write("Overall Statistics:\n")

            avg_confidences = [a['structure_quality'].get('avg_confidence', 0) for a in analyses if a['structure_quality']]
            if avg_confidences:
                f.write(f"Average confidence across all predictions: {np.mean(avg_confidences):.2f} ± {np.std(avg_confidences):.2f}\n")

            base_pair_counts = [a['secondary_structure'].get('num_pairs', 0) for a in analyses if a['secondary_structure']]
            if base_pair_counts:
                f.write(f"Average base pairs: {np.mean(base_pair_counts):.1f} ± {np.std(base_pair_counts):.1f}\n")

def main():
    parser = argparse.ArgumentParser(
        description="RhoFold+ Structure Analysis and Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze a single prediction directory
    python use_case_4_structure_analysis.py --input output/prediction_results/3owzA --output analysis/3owzA_analysis

    # Analyze all predictions in a batch directory
    python use_case_4_structure_analysis.py --input output/batch_results --output analysis/batch_analysis

    # Compare multiple prediction directories
    python use_case_4_structure_analysis.py --input output/prediction_results/3owzA output/prediction_results/5ddoA --output analysis/comparison

    # Analyze with CSV export for further processing
    python use_case_4_structure_analysis.py --input output/batch_results --output analysis/detailed --export_csv

Output files:
    analysis_report.txt     - Comprehensive text report
    analysis_summary.csv    - Summary table (if --export_csv)
    analysis_log.txt        - Execution log
        """
    )

    parser.add_argument(
        "--input", "-i",
        help="Input directory or directories containing RhoFold predictions",
        nargs='+',
        required=True
    )

    parser.add_argument(
        "--output", "-o",
        help="Output directory for analysis results",
        default="analysis_results",
        type=str
    )

    parser.add_argument(
        "--export_csv",
        help="Export summary table as CSV file",
        action="store_true"
    )

    parser.add_argument(
        "--include_unrelaxed",
        help="Prefer unrelaxed structures over relaxed ones for analysis",
        action="store_true"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.output)

    logger.info("=" * 70)
    logger.info("RhoFold+ Structure Analysis and Validation")
    logger.info("=" * 70)

    # Find all prediction outputs
    all_outputs = []
    for input_path in args.input:
        logger.info(f"Scanning: {input_path}")
        outputs = find_prediction_outputs(input_path)
        all_outputs.extend(outputs)
        logger.info(f"Found {len(outputs)} predictions")

    if not all_outputs:
        logger.error("No prediction outputs found!")
        sys.exit(1)

    logger.info(f"Total predictions to analyze: {len(all_outputs)}")

    # Analyze each prediction
    analyses = []
    for output_info in all_outputs:
        try:
            analysis = analyze_single_prediction(output_info, logger)
            analyses.append(analysis)
        except Exception as e:
            logger.error(f"Failed to analyze {output_info['name']}: {e}")

    logger.info(f"Successfully analyzed {len(analyses)} predictions")

    # Generate report
    report_file = os.path.join(args.output, 'analysis_report.txt')
    generate_report(analyses, report_file)
    logger.info(f"Analysis report saved: {report_file}")

    # Export CSV if requested
    if args.export_csv:
        csv_file = os.path.join(args.output, 'analysis_summary.csv')

        # Prepare data for CSV
        csv_data = []
        for analysis in analyses:
            row = {
                'name': analysis['name'],
                'directory': analysis['directory'],
                'length': analysis['structure_quality'].get('length', None),
                'avg_confidence': analysis['structure_quality'].get('avg_confidence', None),
                'min_confidence': analysis['structure_quality'].get('min_confidence', None),
                'max_confidence': analysis['structure_quality'].get('max_confidence', None),
                'std_confidence': analysis['structure_quality'].get('std_confidence', None),
                'base_pairs': analysis['secondary_structure'].get('num_pairs', None),
                'very_high_conf_percent': analysis['structure_quality'].get('very_high_conf_percent', None),
                'confident_percent': analysis['structure_quality'].get('confident_percent', None),
                'low_conf_percent': analysis['structure_quality'].get('low_conf_percent', None),
                'very_low_conf_percent': analysis['structure_quality'].get('very_low_conf_percent', None),
                'has_relaxed_structure': analysis['files_found'].get('relaxed_pdb', False),
                'has_results_npz': analysis['files_found'].get('results_npz', False),
                'has_secondary_structure': analysis['files_found'].get('secondary_structure_ct', False)
            }
            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
        logger.info(f"CSV summary saved: {csv_file}")

    logger.info("=" * 70)
    logger.info("Analysis completed successfully!")
    logger.info(f"Results saved in: {args.output}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()