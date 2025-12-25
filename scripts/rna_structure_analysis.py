#!/usr/bin/env python3
"""
Script: rna_structure_analysis.py
Description: Clean RNA structure analysis and validation from RhoFold outputs

Original Use Case: examples/use_case_4_structure_analysis.py
Dependencies Removed: Inlined logging setup, file parsing, analysis functions

Usage:
    python scripts/rna_structure_analysis.py --input <input_dirs> --output <output_file>

Example:
    python scripts/rna_structure_analysis.py --input output/batch_results --output analysis/batch_analysis --export_csv
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
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import numpy as np
import pandas as pd

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "log_level": "INFO",
    "export_csv": False,
    "include_unrelaxed": False
}

# ==============================================================================
# Inlined Utility Functions (simplified from repo)
# ==============================================================================
def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration. Inlined from original use case."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('RhoFold_Analysis')
    logger.setLevel(getattr(logging, log_level.upper()))

    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

    # File handler
    file_handler = logging.FileHandler(output_dir / 'analysis_log.txt', mode='w')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(getattr(logging, log_level.upper()))
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def parse_pdb_file(pdb_file: Path) -> Dict[str, Any]:
    """Parse PDB file and extract basic information. Inlined from original."""
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

def analyze_structure_quality(pdb_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze structure quality metrics. Inlined from original."""
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

def load_results_npz(npz_file: Path) -> Dict[str, Any]:
    """Load and analyze RhoFold results.npz file. Inlined from original."""
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

def parse_secondary_structure_ct(ct_file: Path) -> Dict[str, Any]:
    """Parse secondary structure from .ct file. Inlined from original."""
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

def find_prediction_outputs(base_dir: Path) -> List[Dict[str, Any]]:
    """Find RhoFold output files in directory. Inlined from original."""
    outputs = []

    # Look for individual prediction directories
    prediction_dirs = []

    if base_dir.is_dir():
        # Check if this is a single prediction directory
        if (base_dir / 'unrelaxed_model.pdb').exists():
            prediction_dirs.append(base_dir)
        else:
            # Look for subdirectories with predictions
            for item in base_dir.iterdir():
                if item.is_dir() and (item / 'unrelaxed_model.pdb').exists():
                    prediction_dirs.append(item)

    for pred_dir in prediction_dirs:
        name = pred_dir.name
        output_info = {'name': name, 'directory': str(pred_dir)}

        # Find output files
        unrelaxed_pdb = pred_dir / 'unrelaxed_model.pdb'
        relaxed_pdbs = list(pred_dir.glob('relaxed_*_model.pdb'))
        results_npz = pred_dir / 'results.npz'
        ss_ct = pred_dir / 'ss.ct'

        output_info['files'] = {
            'unrelaxed_pdb': str(unrelaxed_pdb) if unrelaxed_pdb.exists() else None,
            'relaxed_pdb': str(relaxed_pdbs[0]) if relaxed_pdbs else None,
            'results_npz': str(results_npz) if results_npz.exists() else None,
            'secondary_structure_ct': str(ss_ct) if ss_ct.exists() else None
        }

        outputs.append(output_info)

    return outputs

def analyze_single_prediction(output_info: Dict[str, Any], config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """Analyze a single prediction output. Inlined from original."""
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

    # Check which files exist
    for file_type, file_path in files.items():
        analysis['files_found'][file_type] = file_path is not None
        if file_path:
            analysis['files_found'][f'{file_type}_path'] = file_path

    try:
        # Analyze structure (prefer relaxed over unrelaxed)
        pdb_file = None
        if files['relaxed_pdb'] and not config['include_unrelaxed']:
            pdb_file = files['relaxed_pdb']
            analysis['structure_source'] = 'relaxed'
        elif files['unrelaxed_pdb']:
            pdb_file = files['unrelaxed_pdb']
            analysis['structure_source'] = 'unrelaxed'

        if pdb_file:
            pdb_data = parse_pdb_file(Path(pdb_file))
            analysis['structure_quality'] = analyze_structure_quality(pdb_data)
            analysis['sequence'] = pdb_data['sequence']

        # Analyze results NPZ file
        if files['results_npz']:
            analysis['results_data'] = load_results_npz(Path(files['results_npz']))

        # Analyze secondary structure
        if files['secondary_structure_ct']:
            analysis['secondary_structure'] = parse_secondary_structure_ct(Path(files['secondary_structure_ct']))

        logger.info(f"✓ {name} analysis completed")

    except Exception as e:
        logger.error(f"✗ {name} analysis failed: {e}")
        analysis['error'] = str(e)

    return analysis

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_structure_analysis(
    input_dirs: List[Union[str, Path]],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for RNA structure analysis and validation.

    Args:
        input_dirs: List of RhoFold prediction output directories
        output_file: Path to save output directory (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - analyses: List of individual analysis results
            - summary: Summary statistics
            - output_dir: Path to output directory
            - metadata: Execution metadata

    Example:
        >>> result = run_structure_analysis(["output/batch_results"], "analysis/results")
        >>> print(result['summary'])
    """
    # Setup
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}
    input_dirs = [Path(d) for d in input_dirs]

    # Setup output directory
    if output_file:
        output_dir = Path(output_file)
    else:
        output_dir = Path("analysis") / "structure_analysis"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir, config.get('log_level', 'INFO'))

    logger.info("=" * 70)
    logger.info("RhoFold+ RNA Structure Analysis and Validation")
    logger.info("=" * 70)
    logger.info(f"Input directories: {[str(d) for d in input_dirs]}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Export CSV: {config['export_csv']}")
    logger.info(f"Include unrelaxed: {config['include_unrelaxed']}")

    # Find all prediction outputs
    all_outputs = []
    for input_dir in input_dirs:
        logger.info(f"Scanning directory: {input_dir}")
        outputs = find_prediction_outputs(input_dir)
        all_outputs.extend(outputs)
        logger.info(f"Found {len(outputs)} predictions in {input_dir}")

    if not all_outputs:
        raise ValueError("No prediction outputs found in input directories!")

    logger.info(f"Total predictions to analyze: {len(all_outputs)}")

    # Analyze each prediction
    analyses = []
    successful = 0
    failed = 0

    for output_info in all_outputs:
        try:
            analysis = analyze_single_prediction(output_info, config, logger)
            analyses.append(analysis)
            if 'error' not in analysis:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Failed to analyze {output_info['name']}: {e}")
            analyses.append({
                'name': output_info['name'],
                'error': str(e)
            })
            failed += 1

    # Generate summary statistics
    valid_analyses = [a for a in analyses if 'error' not in a and a.get('structure_quality')]

    summary = {
        'total_structures': len(all_outputs),
        'successful_analyses': successful,
        'failed_analyses': failed,
        'success_rate': successful / len(all_outputs) * 100 if all_outputs else 0
    }

    if valid_analyses:
        # Collect quality metrics
        confidences = [a['structure_quality']['avg_confidence'] for a in valid_analyses]
        lengths = [a['structure_quality']['length'] for a in valid_analyses]

        summary.update({
            'avg_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'avg_length': float(np.mean(lengths)),
            'min_length': int(np.min(lengths)),
            'max_length': int(np.max(lengths))
        })

        # Count file types
        file_counts = {
            'has_relaxed': sum(1 for a in analyses if a.get('files_found', {}).get('relaxed_pdb', False)),
            'has_unrelaxed': sum(1 for a in analyses if a.get('files_found', {}).get('unrelaxed_pdb', False)),
            'has_results': sum(1 for a in analyses if a.get('files_found', {}).get('results_npz', False)),
            'has_secondary': sum(1 for a in analyses if a.get('files_found', {}).get('secondary_structure_ct', False))
        }
        summary.update(file_counts)

    # Write detailed report
    report_file = output_dir / 'analysis_report.txt'
    with open(report_file, 'w') as f:
        f.write("RhoFold+ Structure Analysis Report\n")
        f.write("=" * 40 + "\n\n")

        f.write("Summary:\n")
        f.write("-" * 10 + "\n")
        for key, value in summary.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.2f}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("Individual Analysis Results:\n")
        f.write("-" * 30 + "\n")
        for analysis in analyses:
            name = analysis['name']
            if 'error' in analysis:
                f.write(f"✗ {name}: {analysis['error']}\n")
            else:
                quality = analysis.get('structure_quality', {})
                if quality:
                    f.write(f"✓ {name}: length={quality['length']}, confidence={quality['avg_confidence']:.2f}\n")
                else:
                    f.write(f"? {name}: No structure quality data\n")

        f.write("\nFile Status Summary:\n")
        f.write("-" * 20 + "\n")
        for analysis in analyses:
            name = analysis['name']
            files = analysis.get('files_found', {})
            status_parts = []
            if files.get('relaxed_pdb'): status_parts.append("relaxed")
            if files.get('unrelaxed_pdb'): status_parts.append("unrelaxed")
            if files.get('results_npz'): status_parts.append("results")
            if files.get('secondary_structure_ct'): status_parts.append("secondary")

            f.write(f"{name}: {', '.join(status_parts) if status_parts else 'no files'}\n")

    logger.info(f"Detailed report saved: {report_file}")

    # Export CSV if requested
    csv_file = None
    if config['export_csv'] and valid_analyses:
        csv_file = output_dir / 'analysis_summary.csv'

        # Prepare data for CSV
        csv_data = []
        for analysis in analyses:
            row = {'name': analysis['name']}

            if 'error' in analysis:
                row['status'] = 'failed'
                row['error'] = analysis['error']
            else:
                row['status'] = 'success'

                # Structure quality metrics
                quality = analysis.get('structure_quality', {})
                for key, value in quality.items():
                    row[f'quality_{key}'] = value

                # File status
                files = analysis.get('files_found', {})
                row['has_relaxed'] = files.get('relaxed_pdb', False)
                row['has_unrelaxed'] = files.get('unrelaxed_pdb', False)
                row['has_results'] = files.get('results_npz', False)
                row['has_secondary'] = files.get('secondary_structure_ct', False)

                # Secondary structure info
                ss = analysis.get('secondary_structure', {})
                if ss:
                    row['ss_num_pairs'] = ss.get('num_pairs', 0)
                    row['ss_length'] = ss.get('length', 0)

                # Results data
                results = analysis.get('results_data', {})
                if results.get('plddt_stats'):
                    plddt = results['plddt_stats']
                    for key, value in plddt.items():
                        row[f'plddt_{key}'] = value

            csv_data.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
        logger.info(f"CSV summary saved: {csv_file}")

    logger.info("=" * 70)
    logger.info("Structure analysis completed!")
    logger.info(f"Results saved in: {output_dir}")
    logger.info(f"Successful: {successful}/{len(all_outputs)} ({summary['success_rate']:.1f}%)")
    logger.info("=" * 70)

    return {
        "analyses": analyses,
        "summary": summary,
        "output_dir": str(output_dir),
        "metadata": {
            "total_structures": len(all_outputs),
            "successful_analyses": successful,
            "failed_analyses": failed,
            "config": config,
            "report_file": str(report_file),
            "csv_file": str(csv_file) if csv_file else None
        }
    }

# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', required=True, nargs='+',
                       help='RhoFold prediction output directories to analyze')
    parser.add_argument('--output', '-o', help='Output directory path')
    parser.add_argument('--config', '-c', help='Config file (JSON)')
    parser.add_argument('--export_csv', action='store_true',
                       help='Export summary as CSV')
    parser.add_argument('--include_unrelaxed', action='store_true',
                       help='Prefer unrelaxed structures over relaxed')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override config with CLI arguments
    overrides = {}
    if args.export_csv:
        overrides['export_csv'] = True
    if args.include_unrelaxed:
        overrides['include_unrelaxed'] = True

    # Run
    try:
        result = run_structure_analysis(
            input_dirs=args.input,
            output_file=args.output,
            config=config,
            **overrides
        )

        print(f"✅ Success: Analyzed {result['metadata']['total_structures']} structures")
        print(f"Success rate: {result['summary']['success_rate']:.1f}%")
        print(f"Results saved in {result['output_dir']}")
        if result['metadata']['csv_file']:
            print(f"CSV exported: {result['metadata']['csv_file']}")
        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()