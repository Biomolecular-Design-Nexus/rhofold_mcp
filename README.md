# RhoFold MCP

> RNA 3D structure prediction and analysis tool using RhoFold+ with MCP integration for Claude Code and other AI assistants.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Local Usage (Scripts)](#local-usage-scripts)
- [MCP Server Installation](#mcp-server-installation)
- [Using with Claude Code](#using-with-claude-code)
- [Using with Gemini CLI](#using-with-gemini-cli)
- [Available Tools](#available-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

RhoFold MCP provides comprehensive RNA 3D structure prediction capabilities through the RhoFold+ deep learning model. This MCP server enables AI assistants like Claude Code to predict RNA structures from single sequences or with Multiple Sequence Alignments (MSA), perform batch processing, and analyze structural outputs.

### Features
- **Single Sequence Prediction**: Fast RNA 3D structure prediction using only sequence input
- **MSA-Enhanced Prediction**: Higher accuracy prediction using evolutionary information
- **Batch Processing**: Efficient processing of multiple RNA sequences
- **Structure Analysis**: Quality assessment and validation of predicted structures
- **Job Management**: Background processing for long-running operations
- **Dual Environment Support**: Legacy RhoFold (Python 3.7) + Modern MCP (Python 3.10)

### Directory Structure
```
./
├── README.md                   # This file
├── env/                        # Python 3.10 MCP environment
├── env_py37/                   # Python 3.7 RhoFold environment
├── src/
│   ├── server.py              # MCP server
│   └── jobs/                  # Job management system
├── scripts/
│   ├── rna_single_sequence_prediction.py  # Single sequence folding
│   ├── rna_msa_prediction.py              # MSA-enhanced folding
│   ├── rna_batch_prediction.py            # Batch processing
│   ├── rna_structure_analysis.py          # Structure analysis
│   └── lib/                               # Shared utilities
├── examples/
│   ├── data/                  # Demo RNA sequences (16 examples)
│   └── use_case_*.py          # Example scripts
├── configs/                   # Configuration templates
├── repo/                      # Original RhoFold repository
└── jobs/                      # Background job outputs
```

---

## Installation

### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.10+ (for MCP)
- Python 3.7.12 (for RhoFold dependencies)
- ~10GB disk space for environments
- CUDA support optional (5-10x speedup)

### Create Environment
Please strictly following the information in `reports/step3_environment.md` to obtain the procedure to setup the environment. An example workflow is shown below.

```bash
# Navigate to the MCP directory
cd /home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/rhofold_mcp

# Create conda environment (use mamba if available)
mamba create -p ./env python=3.10 -y
# or: conda create -p ./env python=3.10 -y

# Activate environment
mamba activate ./env
# or: conda activate ./env

# Install MCP dependencies
pip install fastmcp loguru pandas numpy tqdm

# Create RhoFold environment
mamba env create -f repo/RhoFold/envs/environment_linux.yaml -p ./env_py37

# Install RhoFold package
mamba run -p ./env_py37 pip install -e repo/RhoFold/
```

---

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Example |
|--------|-------------|---------|
| `scripts/rna_single_sequence_prediction.py` | Single sequence RNA 3D structure prediction | See below |
| `scripts/rna_msa_prediction.py` | MSA-enhanced RNA structure prediction | See below |
| `scripts/rna_batch_prediction.py` | Batch processing of multiple RNA sequences | See below |
| `scripts/rna_structure_analysis.py` | Analyze and validate predicted RNA structures | See below |

### Script Examples

#### Single Sequence Prediction

```bash
# Activate environment
mamba activate ./env_py37

# Run script
python scripts/rna_single_sequence_prediction.py \
  --input examples/data/3owzA/3owzA.fasta \
  --output results/single_prediction \
  --device cpu \
  --relax_steps 1000
```

**Parameters:**
- `--input, -i`: Path to RNA sequence in FASTA format (required)
- `--output, -o`: Output directory for prediction results (default: auto-generated)
- `--device`: Computation device - cpu, cuda:0, etc. (default: cpu)
- `--relax_steps`: Number of Amber relaxation steps (default: 1000)

#### MSA-Enhanced Prediction

```bash
python scripts/rna_msa_prediction.py \
  --fasta examples/data/3owzA/3owzA.fasta \
  --msa examples/data/3owzA/3owzA.a3m \
  --output results/msa_prediction \
  --device cpu
```

**Parameters:**
- `--fasta, -f`: Path to RNA sequence in FASTA format (required)
- `--msa, -m`: Path to Multiple Sequence Alignment in A3M format (required)
- `--output, -o`: Output directory for results (default: auto-generated)
- `--device`: Computation device (default: cpu)
- `--relax_steps`: Amber relaxation steps (default: 1000)

#### Batch Processing

```bash
python scripts/rna_batch_prediction.py \
  --input_dir examples/data \
  --output results/batch_results \
  --max_sequences 3 \
  --device cpu
```

**Parameters:**
- `--input_dir`: Directory containing FASTA files (and optional A3M files)
- `--multi_fasta`: Alternative: single multi-FASTA file
- `--output, -o`: Output directory for all results (default: auto-generated)
- `--max_sequences`: Maximum number of sequences to process
- `--include`: Comma-separated list of sequence names to include
- `--exclude`: Comma-separated list of sequence names to exclude

#### Structure Analysis

```bash
# Activate MCP environment for analysis
mamba activate ./env

python scripts/rna_structure_analysis.py \
  --input results/batch_results \
  --output analysis/comprehensive_analysis \
  --export_csv
```

**Parameters:**
- `--input, -i`: Directory containing RhoFold prediction results (required)
- `--output, -o`: Output directory for analysis results (default: auto-generated)
- `--export_csv`: Export summary as CSV format (flag)
- `--include_unrelaxed`: Include unrelaxed structures in analysis (flag)

---

## MCP Server Installation

### Option 1: Using fastmcp (Recommended)

```bash
# Install MCP server for Claude Code
fastmcp install src/server.py --name RhoFold
```

### Option 2: Manual Installation for Claude Code

```bash
# Add MCP server to Claude Code
claude mcp add RhoFold -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify installation
claude mcp list
```

### Option 3: Configure in settings.json

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "RhoFold": {
      "command": "/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/rhofold_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/rhofold_mcp/src/server.py"]
    }
  }
}
```

---

## Using with Claude Code

After installing the MCP server, you can use it directly in Claude Code.

### Quick Start

```bash
# Start Claude Code
claude
```

### Example Prompts

#### Tool Discovery
```
What tools are available from RhoFold?
```

#### Basic Usage
```
Use predict_rna_structure with input file @examples/data/3owzA/3owzA.fasta
```

#### MSA-Enhanced Prediction
```
Run predict_rna_structure_with_msa on @examples/data/3owzA/3owzA.fasta using MSA @examples/data/3owzA/3owzA.a3m
```

#### Long-Running Tasks (Submit API)
```
Submit batch RNA prediction for @examples/data/ directory with max 3 sequences
Then check the job status
```

#### Batch Processing
```
Process these files in batch:
- @examples/data/3owzA/3owzA.fasta
- @examples/data/4l81A/4l81A.fasta
- @examples/data/5ddoA/5ddoA.fasta
```

### Using @ References

In Claude Code, use `@` to reference files and directories:

| Reference | Description |
|-----------|-------------|
| `@examples/data/3owzA/3owzA.fasta` | Reference a specific FASTA file |
| `@examples/data/3owzA/3owzA.a3m` | Reference a specific MSA file |
| `@configs/default.json` | Reference a config file |
| `@results/` | Reference output directory |

---

## Using with Gemini CLI

### Configuration

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "RhoFold": {
      "command": "/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/rhofold_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/rhofold_mcp/src/server.py"]
    }
  }
}
```

### Example Prompts

```bash
# Start Gemini CLI
gemini

# Example prompts (same as Claude Code)
> What tools are available?
> Use predict_rna_structure with file examples/data/3owzA/3owzA.fasta
```

---

## Available Tools

### Quick Operations (Sync API)

These tools return results immediately (< 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `predict_rna_structure` | Single sequence RNA 3D structure prediction | `input_file`, `output_file`, `device`, `relax_steps` |
| `predict_rna_structure_with_msa` | MSA-enhanced RNA structure prediction | `fasta_file`, `msa_file`, `output_file`, `device`, `relax_steps` |
| `analyze_rna_structures` | Analyze and validate predicted RNA structures | `input_dir`, `output_file`, `export_csv`, `include_unrelaxed` |
| `validate_rna_sequence` | Validate RNA sequence file format | `sequence_file` |
| `get_example_data_info` | List available example datasets | None |

### Long-Running Tasks (Submit API)

These tools return a job_id for tracking (> 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `submit_batch_rna_prediction` | Batch processing of multiple RNA sequences | `input_dir`, `multi_fasta`, `output_dir`, `max_sequences`, `device`, `job_name` |
| `submit_large_sequence_prediction` | Single sequence prediction (background) | `input_file`, `output_dir`, `device`, `relax_steps`, `job_name` |
| `submit_msa_prediction` | MSA-based prediction (background) | `fasta_file`, `msa_file`, `output_dir`, `device`, `relax_steps`, `job_name` |
| `submit_comprehensive_analysis` | Large-scale structure analysis | `prediction_results_dir`, `output_dir`, `job_name` |

### Job Management Tools

| Tool | Description |
|------|-------------|
| `get_job_status` | Check job progress |
| `get_job_result` | Get results when completed |
| `get_job_log` | View execution logs |
| `cancel_job` | Cancel running job |
| `list_jobs` | List all jobs |

---

## Examples

### Example 1: Single Sequence RNA Structure Prediction

**Goal:** Predict 3D structure of a single RNA sequence

**Using Script:**
```bash
mamba run -p ./env_py37 python scripts/rna_single_sequence_prediction.py \
  --input examples/data/3owzA/3owzA.fasta \
  --output results/example1/
```

**Using MCP (in Claude Code):**
```
Use predict_rna_structure to process @examples/data/3owzA/3owzA.fasta and save results to results/example1/
```

**Expected Output:**
- `unrelaxed_model.pdb`: Raw 3D structure prediction
- `relaxed_model.pdb`: Amber-refined structure
- `ss.ct`: Secondary structure in CT format
- `results.npz`: Distance maps and confidence scores

### Example 2: MSA-Enhanced Structure Prediction

**Goal:** Use evolutionary information for higher accuracy prediction

**Using Script:**
```bash
mamba run -p ./env_py37 python scripts/rna_msa_prediction.py \
  --fasta examples/data/3owzA/3owzA.fasta \
  --msa examples/data/3owzA/3owzA.a3m \
  --output results/example2/
```

**Using MCP (in Claude Code):**
```
Use predict_rna_structure_with_msa on @examples/data/3owzA/3owzA.fasta with MSA @examples/data/3owzA/3owzA.a3m
```

### Example 3: Batch Processing

**Goal:** Process multiple files at once

**Using Script:**
```bash
for f in examples/data/*/**.fasta; do
  mamba run -p ./env_py37 python scripts/rna_single_sequence_prediction.py --input "$f" --output results/batch/
done
```

**Using MCP (in Claude Code):**
```
Submit batch processing for all FASTA files in @examples/data/
```

---

## Demo Data

The `examples/data/` directory contains sample data for testing:

| File | Description | Use With |
|------|-------------|----------|
| `3owzA/3owzA.fasta` | 88-nucleotide RNA structure with MSA | All prediction tools |
| `3owzB/3owzB.fasta` | Alternative conformation of 3owzA | MSA comparison studies |
| `4l81A/4l81A.fasta` | Large ribosomal RNA fragment | Batch processing |
| `5ddoA/5ddoA.fasta` | Structured RNA domain | Structure analysis |
| `...` | 12 additional RNA examples | Various RNA types |

**Total Demo Sequences**: 16 RNA structures with corresponding FASTA and A3M files

---

## Configuration Files

The `configs/` directory contains configuration templates:

| Config | Description | Parameters |
|--------|-------------|------------|
| `default.json` | General purpose settings | device, relax_steps, output_format |
| `single_sequence_prediction.json` | Single sequence specific settings | device, relax_steps |
| `msa_prediction.json` | MSA-enhanced prediction settings | device, relax_steps, msa_params |
| `batch_prediction.json` | Batch processing configuration | max_sequences, parallel_jobs, device |
| `structure_analysis.json` | Analysis and validation settings | export_formats, quality_metrics |

### Config Example

```json
{
  "device": "cpu",
  "relax_steps": 1000,
  "output_format": "pdb",
  "quality_checks": true,
  "save_intermediate": false
}
```

---

## Troubleshooting

### Environment Issues

**Problem:** Environment not found
```bash
# Recreate environment
mamba create -p ./env python=3.10 -y
mamba activate ./env
pip install fastmcp loguru pandas numpy tqdm
```

**Problem:** Import errors
```bash
# Verify installation
python -c "from src.server import mcp"
```

### MCP Issues

**Problem:** Server not found in Claude Code
```bash
# Check MCP registration
claude mcp list

# Re-add if needed
claude mcp remove RhoFold
claude mcp add RhoFold -- $(pwd)/env/bin/python $(pwd)/src/server.py
```

**Problem:** Tools not working
```bash
# Test server directly
python -c "
from src.server import mcp
print('MCP server loaded successfully')
"
```

### Job Issues

**Problem:** Job stuck in pending
```bash
# Check job directory
ls -la jobs/

# View job log
cat jobs/<job_id>/job.log
```

**Problem:** Job failed
```
Use get_job_log with job_id "<job_id>" and tail 100 to see error details
```

### Performance Issues

**Problem:** Slow execution
- **Solution**: Use GPU with `device: "cuda:0"` for 5-10x speedup
- **Alternative**: Reduce `relax_steps` from 1000 to 100 for faster results

**Problem:** Out of memory
- **Solution**: Process fewer sequences at once in batch mode
- **Monitor**: Peak memory usage is ~4-6GB per sequence

---

## Development

### Running Tests

```bash
# Activate environment
mamba activate ./env

# Run tests
pytest tests/ -v
```

### Starting Dev Server

```bash
# Run MCP server in dev mode
fastmcp dev src/server.py
```

---

## Performance Guidelines

### Execution Time Estimates

| Operation | CPU Runtime | GPU Runtime | Memory Usage |
|-----------|-------------|-------------|--------------|
| Single sequence (88 nt) | ~5 minutes | ~30 seconds | ~4-6 GB |
| MSA prediction (88 nt) | ~5 minutes | ~30 seconds | ~4-6 GB |
| Batch (3 sequences) | ~15 minutes | ~2 minutes | ~4-6 GB |
| Structure analysis | ~10 seconds | N/A | ~100 MB |

### Optimization Tips

1. **GPU Usage**: Add `device: "cuda:0"` for 5-10x speedup
2. **Batch Processing**: More efficient than multiple single predictions
3. **Relaxation Steps**: Reduce from 1000 to 100 for faster results (slight quality loss)
4. **Memory Management**: Process large batches in smaller chunks using `max_sequences`

### Resource Planning

- **Storage**: Expect ~4 MB per predicted structure
- **CPU**: High usage during model inference (~5 min per sequence)
- **Memory**: Peak usage ~4-6 GB during prediction
- **Network**: 508 MB model download on first use
- **GPU**: CUDA-compatible GPU provides 5-10x speedup

---

## License

This MCP server is based on [RhoFold](https://github.com/RFOLD/RhoFold) and distributed under the same license terms.

## Credits

- **RhoFold**: Original RNA structure prediction model by the RhoFold team
- **FastMCP**: MCP framework for AI assistant integration
- **Original Repository**: [RhoFold GitHub](https://github.com/RFOLD/RhoFold)

---

## Citation

If you use RhoFold MCP in your research, please cite the original RhoFold paper:

```
@article{shen2022rhofold,
  title={RNA structure prediction using deep learning},
  author={Shen, Tianyu and others},
  journal={Nature Methods},
  year={2022}
}
```

For questions or support, please refer to the [original RhoFold repository](https://github.com/RFOLD/RhoFold) or create an issue in the MCP repository.