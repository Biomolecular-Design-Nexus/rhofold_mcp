# Step 3: Use Cases Report

## Scan Information
- **Scan Date**: 2024-12-24
- **Filter Applied**: RNA 3D structure prediction, RNA embedding generation, RNA secondary structure prediction from 3D, single-sequence RNA structure prediction
- **Python Version**: 3.7.12 (legacy) + 3.10.19 (MCP)
- **Environment Strategy**: Dual environment setup

## Use Cases

### UC-001: Single Sequence RNA 3D Structure Prediction
- **Description**: Fast RNA 3D structure prediction using only sequence input (no MSA)
- **Script Path**: `examples/use_case_1_single_sequence_prediction.py`
- **Complexity**: Simple
- **Priority**: High
- **Environment**: `./env_py37`
- **Source**: `repo/RhoFold/README.md` section "Folding with single sequence as input"

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| input_fasta | file | RNA sequence in FASTA format | --input, -i |
| output_dir | directory | Output directory for results | --output, -o |
| device | string | Computation device (cpu/cuda) | --device |
| relax_steps | integer | Amber relaxation iterations | --relax_steps |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| unrelaxed_model.pdb | file | Raw 3D structure prediction |
| relaxed_N_model.pdb | file | Amber-refined structure |
| ss.ct | file | Secondary structure (CT format) |
| results.npz | file | Distance maps and confidence scores |
| log.txt | file | Execution log |

**Example Usage:**
```bash
mamba run -p ./env_py37 python examples/use_case_1_single_sequence_prediction.py --input examples/data/3owzA/3owzA.fasta --output output/3owzA_single
```

**Example Data**: `examples/data/3owzA/3owzA.fasta`

---

### UC-002: RNA 3D Structure Prediction with MSA
- **Description**: High-accuracy RNA 3D structure prediction using sequence + Multiple Sequence Alignment
- **Script Path**: `examples/use_case_2_prediction_with_msa.py`
- **Complexity**: Medium
- **Priority**: High
- **Environment**: `./env_py37`
- **Source**: `repo/RhoFold/README.md` section "Folding with sequence and given MSA as input"

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| fasta_file | file | RNA sequence in FASTA format | --fasta, -f |
| msa_file | file | Multiple sequence alignment (A3M format) | --msa, -m |
| output_dir | directory | Output directory for results | --output, -o |
| device | string | Computation device (cpu/cuda) | --device |
| relax_steps | integer | Amber relaxation iterations | --relax_steps |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| unrelaxed_model.pdb | file | Raw 3D structure prediction |
| relaxed_N_model.pdb | file | Amber-refined structure |
| ss.ct | file | Secondary structure (CT format) |
| results.npz | file | Distance maps and confidence scores |
| log.txt | file | Execution log |

**Example Usage:**
```bash
mamba run -p ./env_py37 python examples/use_case_2_prediction_with_msa.py --fasta examples/data/3owzA/3owzA.fasta --msa examples/data/3owzA/3owzA.a3m --output output/3owzA_msa
```

**Example Data**: `examples/data/3owzA/3owzA.fasta` + `examples/data/3owzA/3owzA.a3m`

---

### UC-003: Batch RNA 3D Structure Prediction
- **Description**: Process multiple RNA sequences efficiently in batch mode
- **Script Path**: `examples/use_case_3_batch_prediction.py`
- **Complexity**: Complex
- **Priority**: High
- **Environment**: `./env_py37`
- **Source**: Custom implementation based on RhoFold inference pipeline

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| input_dir | directory | Directory with FASTA/A3M files | --input_dir |
| multi_fasta | file | Multi-FASTA file with sequences | --multi_fasta |
| output_dir | directory | Output directory for batch results | --output, -o |
| include | string | Comma-separated sequence names to include | --include |
| exclude | string | Comma-separated sequence names to exclude | --exclude |
| max_sequences | integer | Maximum sequences to process | --max_sequences |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| batch_log.txt | file | Overall batch execution log |
| batch_summary.txt | file | Summary of all predictions |
| sequence_N/ | directories | Individual prediction results |

**Example Usage:**
```bash
mamba run -p ./env_py37 python examples/use_case_3_batch_prediction.py --input_dir examples/data --output output/batch_results
```

**Example Data**: All directories in `examples/data/` (16 RNA examples)

---

### UC-004: RNA Structure Analysis and Validation
- **Description**: Analyze predicted RNA structures, calculate quality metrics, and generate reports
- **Script Path**: `examples/use_case_4_structure_analysis.py`
- **Complexity**: Medium
- **Priority**: Medium
- **Environment**: `./env` (uses pandas, no RhoFold dependency)
- **Source**: Custom analysis tool for RhoFold outputs

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| input_dirs | directories | RhoFold prediction output directories | --input, -i |
| output_dir | directory | Analysis results directory | --output, -o |
| export_csv | flag | Export summary as CSV | --export_csv |
| include_unrelaxed | flag | Prefer unrelaxed structures | --include_unrelaxed |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| analysis_report.txt | file | Comprehensive text report |
| analysis_summary.csv | file | Summary table (if --export_csv) |
| analysis_log.txt | file | Analysis execution log |

**Example Usage:**
```bash
mamba run -p ./env python examples/use_case_4_structure_analysis.py --input output/batch_results --output analysis/batch_analysis --export_csv
```

**Example Data**: Output from use cases 1-3

---

## Summary

| Metric | Count |
|--------|-------|
| Total Use Cases Found | 4 |
| Scripts Created | 4 |
| High Priority | 3 |
| Medium Priority | 1 |
| Low Priority | 0 |
| Demo Data Copied | ✅ |

## Demo Data Index

| Source | Destination | Description |
|--------|-------------|-------------|
| `repo/RhoFold/example/input/3owzA/` | `examples/data/3owzA/` | RNA sequence and MSA for 3owzA structure |
| `repo/RhoFold/example/input/5ddoA/` | `examples/data/5ddoA/` | RNA sequence and MSA for 5ddoA structure |
| `repo/RhoFold/example/input/4xw7A/` | `examples/data/4xw7A/` | RNA sequence and MSA for 4xw7A structure |
| `repo/RhoFold/example/input/3meiA/` | `examples/data/3meiA/` | RNA sequence and MSA for 3meiA structure |
| `repo/RhoFold/example/input/3owzB/` | `examples/data/3owzB/` | RNA sequence and MSA for 3owzB structure |
| `repo/RhoFold/example/input/4l81A/` | `examples/data/4l81A/` | RNA sequence and MSA for 4l81A structure |
| `repo/RhoFold/example/input/5k7cA/` | `examples/data/5k7cA/` | RNA sequence and MSA for 5k7cA structure |
| `repo/RhoFold/example/input/5kpyA/` | `examples/data/5kpyA/` | RNA sequence and MSA for 5kpyA structure |
| `repo/RhoFold/example/input/5lysA/` | `examples/data/5lysA/` | RNA sequence and MSA for 5lysA structure |
| `repo/RhoFold/example/input/5nwqA/` | `examples/data/5nwqA/` | RNA sequence and MSA for 5nwqA structure |
| `repo/RhoFold/example/input/5tpyA/` | `examples/data/5tpyA/` | RNA sequence and MSA for 5tpyA structure |
| `repo/RhoFold/example/input/6e8uB/` | `examples/data/6e8uB/` | RNA sequence and MSA for 6e8uB structure |
| `repo/RhoFold/example/input/6jq5A/` | `examples/data/6jq5A/` | RNA sequence and MSA for 6jq5A structure |
| `repo/RhoFold/example/input/6p2hA/` | `examples/data/6p2hA/` | RNA sequence and MSA for 6p2hA structure |
| `repo/RhoFold/example/input/6tb7A/` | `examples/data/6tb7A/` | RNA sequence and MSA for 6tb7A structure |
| `repo/RhoFold/example/input/7elpA/` | `examples/data/7elpA/` | RNA sequence and MSA for 7elpA structure |

**Total Demo Sequences**: 16 RNA structures with corresponding FASTA and A3M files

## Use Case Classification

### By Complexity:
- **Simple**: 1 use case (single sequence prediction)
- **Medium**: 2 use cases (MSA prediction, structure analysis)
- **Complex**: 1 use case (batch processing)

### By Function:
- **Primary Prediction**: 3 use cases (single, MSA, batch)
- **Analysis/Validation**: 1 use case (structure analysis)

### By Input Type:
- **Single Sequence**: 1 use case
- **Sequence + MSA**: 1 use case
- **Multiple Sequences**: 1 use case
- **Structure Files**: 1 use case

All use cases are well-documented with example commands, input/output specifications, and test data. The scripts include comprehensive error handling, logging, and validation to ensure robust operation.