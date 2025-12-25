# RhoFold+ Examples and Demo Data

This directory contains example scripts demonstrating RhoFold+ functionality and demo data for testing.

## Use Case Scripts

### 1. Single Sequence Prediction (`use_case_1_single_sequence_prediction.py`)
Fast RNA 3D structure prediction from sequence alone (no MSA required).

**Usage:**
```bash
mamba run -p ../env_py37 python use_case_1_single_sequence_prediction.py \
    --input data/3owzA/3owzA.fasta \
    --output ../output/3owzA_single \
    --device cpu
```

**Features:**
- Fastest prediction mode
- No MSA generation required
- Good for quick approximate structures
- RNA sequence validation
- Automatic model download

### 2. MSA-based Prediction (`use_case_2_prediction_with_msa.py`)
High-accuracy RNA 3D structure prediction using sequence + MSA.

**Usage:**
```bash
mamba run -p ../env_py37 python use_case_2_prediction_with_msa.py \
    --fasta data/3owzA/3owzA.fasta \
    --msa data/3owzA/3owzA.a3m \
    --output ../output/3owzA_msa \
    --device cpu
```

**Features:**
- Highest accuracy mode
- Uses evolutionary information
- MSA quality validation
- Confidence scoring

### 3. Batch Prediction (`use_case_3_batch_prediction.py`)
Process multiple RNA sequences efficiently.

**Usage:**
```bash
# Process all demo data
mamba run -p ../env_py37 python use_case_3_batch_prediction.py \
    --input_dir data \
    --output ../output/batch_results \
    --max_sequences 5

# Process multi-FASTA file
mamba run -p ../env_py37 python use_case_3_batch_prediction.py \
    --multi_fasta my_sequences.fasta \
    --output ../output/multi_results
```

**Features:**
- Batch processing of multiple sequences
- Progress tracking and logging
- Error handling per sequence
- Summary reports

### 4. Structure Analysis (`use_case_4_structure_analysis.py`)
Analyze and validate predicted RNA structures.

**Usage:**
```bash
mamba run -p ../env python use_case_4_structure_analysis.py \
    --input ../output/batch_results \
    --output ../analysis/batch_analysis \
    --export_csv
```

**Features:**
- Quality metrics calculation
- Confidence score analysis
- Structure comparison
- CSV export for further analysis

## Demo Data (`data/` directory)

Contains 16 RNA structures with sequence and MSA files:

| ID | Description | Length | Files |
|----|-------------|--------|-------|
| 3owzA | RNA structure example | 86 nt | .fasta, .a3m |
| 5ddoA | RNA structure example | Variable | .fasta, .a3m |
| 4xw7A | RNA structure example | Variable | .fasta, .a3m |
| 3meiA | RNA structure example | Variable | .fasta, .a3m |
| 3owzB | RNA structure example | Variable | .fasta, .a3m |
| 4l81A | RNA structure example | Variable | .fasta, .a3m |
| 5k7cA | RNA structure example | Variable | .fasta, .a3m |
| 5kpyA | RNA structure example | Variable | .fasta, .a3m |
| 5lysA | RNA structure example | Variable | .fasta, .a3m |
| 5nwqA | RNA structure example | Variable | .fasta, .a3m |
| 5tpyA | RNA structure example | Variable | .fasta, .a3m |
| 6e8uB | RNA structure example | Variable | .fasta, .a3m |
| 6jq5A | RNA structure example | Variable | .fasta, .a3m |
| 6p2hA | RNA structure example | Variable | .fasta, .a3m |
| 6tb7A | RNA structure example | Variable | .fasta, .a3m |
| 7elpA | RNA structure example | Variable | .fasta, .a3m |

### File Formats

**FASTA files (`.fasta`):**
```
>3owzA
GGCUCUGGAGAGAACCGUUUAAUCGGUCGCCGAAGGAGCAAGCUCUGCGGAAACGCAGAGUGAAACUCUCAGGCAAAAGGACAGAGUC
```

**MSA files (`.a3m`):**
Multiple sequence alignment in A3M format, containing the target sequence and homologous sequences.

## Quick Start Examples

### Test Single Sequence Prediction
```bash
cd examples
mamba run -p ../env_py37 python use_case_1_single_sequence_prediction.py \
    --input data/3owzA/3owzA.fasta \
    --output ../test_output/single_test
```

### Test MSA-based Prediction
```bash
cd examples
mamba run -p ../env_py37 python use_case_2_prediction_with_msa.py \
    --fasta data/3owzA/3owzA.fasta \
    --msa data/3owzA/3owzA.a3m \
    --output ../test_output/msa_test
```

### Test Batch Processing (Limited)
```bash
cd examples
mamba run -p ../env_py37 python use_case_3_batch_prediction.py \
    --input_dir data \
    --output ../test_output/batch_test \
    --max_sequences 3 \
    --relax_steps 0  # Skip relaxation for faster testing
```

### Test Structure Analysis
```bash
cd examples
mamba run -p ../env python use_case_4_structure_analysis.py \
    --input ../test_output \
    --output ../test_analysis \
    --export_csv
```

## Output Structure

After running predictions, expect this output structure:

```
output/
├── prediction_name/
│   ├── unrelaxed_model.pdb    # Raw prediction
│   ├── relaxed_1000_model.pdb # Refined structure
│   ├── ss.ct                  # Secondary structure
│   ├── results.npz            # Distance maps & confidence
│   └── log.txt                # Execution log
└── batch_summary.txt          # Batch results summary
```

## Performance Notes

- **Single sequence**: ~30 seconds per sequence (CPU)
- **MSA-based**: ~60 seconds per sequence (CPU)
- **GPU acceleration**: 5-10x faster with CUDA
- **Memory usage**: 2-8GB RAM depending on sequence length
- **Model download**: ~500MB on first run

## Troubleshooting

### Common Issues

1. **Environment errors**: Make sure to use the correct environment
   - RhoFold scripts: `mamba run -p ../env_py37`
   - Analysis scripts: `mamba run -p ../env`

2. **Memory errors**: Use shorter sequences or CPU mode

3. **Missing model**: First run will download the model automatically

4. **Permission errors**: Ensure output directories are writable

### Getting Help

For script-specific help:
```bash
python use_case_N_script.py --help
```

For detailed documentation, see the main README.md in the parent directory.