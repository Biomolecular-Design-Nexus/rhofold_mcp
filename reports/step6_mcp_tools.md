# Step 6: MCP Tools Documentation

## Server Information
- **Server Name**: RhoFold
- **Version**: 1.0.0
- **Created Date**: 2024-12-24
- **Server Path**: `src/server.py`

## Job Management Tools

| Tool | Description | Usage |
|------|-------------|-------|
| `get_job_status` | Check job progress and current status | Monitor background jobs |
| `get_job_result` | Get completed job results and output files | Retrieve finished job outputs |
| `get_job_log` | View job execution logs (last 50 lines by default) | Debug job issues |
| `cancel_job` | Cancel running job | Stop long-running operations |
| `list_jobs` | List all jobs, optionally filtered by status | Job queue management |

## Synchronous Tools (Fast Operations < 10 min)

| Tool | Description | Source Script | Est. Runtime | Input Types |
|------|-------------|---------------|--------------|-------------|
| `predict_rna_structure` | Single sequence RNA 3D structure prediction | `scripts/rna_single_sequence_prediction.py` | ~5 min | FASTA file |
| `predict_rna_structure_with_msa` | MSA-enhanced RNA structure prediction | `scripts/rna_msa_prediction.py` | ~5 min | FASTA + A3M files |
| `analyze_rna_structures` | Analyze and validate RNA structures | `scripts/rna_structure_analysis.py` | ~10 sec | RhoFold output directory |

### Tool Details

#### predict_rna_structure
- **Description**: Predict RNA 3D structure from single sequence using RhoFold
- **Source Script**: `scripts/rna_single_sequence_prediction.py`
- **Estimated Runtime**: ~5 minutes (CPU), ~30 seconds (GPU)

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| input_file | str | Yes | - | Path to RNA sequence in FASTA format |
| output_file | str | No | None | Optional output directory path |
| device | str | No | cpu | Device for computation (cpu, cuda:0, etc.) |
| relax_steps | int | No | 1000 | Number of Amber relaxation steps |

**Returns:**
- `unrelaxed_structure`: Raw 3D structure PDB file
- `relaxed_structure`: Amber-relaxed PDB file (if relax_steps > 0)
- `secondary_structure`: Secondary structure in CT format
- `distance_maps`: NPZ file with distance maps and confidence scores

**Example:**
```
Use predict_rna_structure with input_file "examples/data/3owzA/3owzA.fasta"
```

#### predict_rna_structure_with_msa
- **Description**: Enhanced RNA structure prediction using Multiple Sequence Alignment
- **Source Script**: `scripts/rna_msa_prediction.py`
- **Estimated Runtime**: ~5 minutes (with better accuracy than single sequence)

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| fasta_file | str | Yes | - | Path to RNA sequence in FASTA format |
| msa_file | str | Yes | - | Path to MSA file in A3M format |
| output_file | str | No | None | Optional output directory path |
| device | str | No | cpu | Device for computation |
| relax_steps | int | No | 1000 | Number of Amber relaxation steps |

**Example:**
```
Use predict_rna_structure_with_msa with fasta_file "examples/data/3owzA/3owzA.fasta" and msa_file "examples/data/3owzA/3owzA.a3m"
```

#### analyze_rna_structures
- **Description**: Comprehensive analysis and validation of RhoFold prediction outputs
- **Source Script**: `scripts/rna_structure_analysis.py`
- **Estimated Runtime**: ~10 seconds

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| input_dir | str | Yes | - | Directory containing RhoFold prediction results |
| output_file | str | No | None | Optional output directory for analysis results |
| export_csv | bool | No | True | Export results as CSV format |
| include_unrelaxed | bool | No | False | Analyze unrelaxed structures too |

**Example:**
```
Use analyze_rna_structures with input_dir "output/batch_results"
```

---

## Submit Tools (Long Operations > 10 min)

| Tool | Description | Source Script | Est. Runtime | Batch Support |
|------|-------------|---------------|--------------|---------------|
| `submit_batch_rna_prediction` | Batch processing of multiple RNA sequences | `scripts/rna_batch_prediction.py` | >10 min | ✅ Yes |
| `submit_large_sequence_prediction` | Single sequence prediction (background) | `scripts/rna_single_sequence_prediction.py` | Variable | ❌ No |
| `submit_msa_prediction` | MSA-based prediction (background) | `scripts/rna_msa_prediction.py` | Variable | ❌ No |
| `submit_comprehensive_analysis` | Large-scale structure analysis | `scripts/rna_structure_analysis.py` | Variable | ✅ Yes |

### Tool Details

#### submit_batch_rna_prediction
- **Description**: Process multiple RNA sequences in batch mode with optional MSA enhancement
- **Source Script**: `scripts/rna_batch_prediction.py`
- **Estimated Runtime**: 4+ minutes per sequence (depends on batch size)
- **Supports Batch**: ✅ Yes

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| input_dir | str | No | None | Directory containing FASTA (and optional A3M) files |
| multi_fasta | str | No | None | Multi-FASTA file (alternative to input_dir) |
| output_dir | str | No | auto | Directory to save all prediction outputs |
| max_sequences | int | No | None | Maximum sequences to process (safety limit) |
| device | str | No | cpu | Device for computation |
| relax_steps | int | No | 1000 | Amber relaxation steps per structure |
| job_name | str | No | auto | Custom job name for tracking |

**Example:**
```
Submit batch RNA prediction for examples/data directory with max 5 sequences
```

#### submit_large_sequence_prediction
- **Description**: Background processing for very large RNA sequences
- **Use Case**: Sequences that may take more than 10 minutes or background execution
- **Supports Resume**: ✅ Yes (via job management)

**Example:**
```
Submit large sequence prediction for very_large_rna.fasta
```

#### submit_msa_prediction
- **Description**: Background MSA-based prediction for large datasets
- **Use Case**: Large MSAs or when background execution is preferred

**Example:**
```
Submit MSA prediction for large_sequence.fasta with large_msa.a3m
```

---

## Utility Tools

| Tool | Description | Runtime |
|------|-------------|---------|
| `validate_rna_sequence` | Validate FASTA file format and RNA sequence | ~1 second |
| `get_example_data_info` | List available example datasets | ~1 second |

### Tool Details

#### validate_rna_sequence
- **Description**: Validate RNA sequence file format and nucleotide composition
- **Runtime**: ~1 second

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| sequence_file | str | Yes | Path to FASTA file to validate |

**Returns:**
- Validation status (valid/invalid)
- Sequence length
- Sequence preview (first 50 nucleotides)
- Error details (if invalid)

#### get_example_data_info
- **Description**: Information about available example datasets for testing
- **Returns**: List of example sequences with file paths and metadata

---

## Workflow Examples

### Quick Single Sequence Analysis (Sync)
```
1. Validate: Use validate_rna_sequence with sequence_file "input.fasta"
   → Returns: {"status": "success", "sequence_length": 88, "valid": true}

2. Predict: Use predict_rna_structure with input_file "input.fasta"
   → Returns: Results immediately (~5 min execution)

3. Analyze: Use analyze_rna_structures with input_dir "output/prediction"
   → Returns: Structure analysis and validation results
```

### MSA-Enhanced Prediction (Sync)
```
1. Predict: Use predict_rna_structure_with_msa with fasta_file "seq.fasta" and msa_file "seq.a3m"
   → Returns: Enhanced prediction with better confidence scores

2. Analyze: Use analyze_rna_structures with input_dir "output/msa_prediction"
   → Returns: Quality metrics showing improvement over single sequence
```

### Large-Scale Batch Processing (Async)
```
1. Submit: Use submit_batch_rna_prediction with input_dir "data/sequences" and max_sequences 10
   → Returns: {"job_id": "abc123", "status": "submitted"}

2. Monitor: Use get_job_status with job_id "abc123"
   → Returns: {"status": "running", "progress": "Processing sequence 3/10"}

3. Check logs: Use get_job_log with job_id "abc123"
   → Returns: Real-time execution logs

4. Get results: Use get_job_result with job_id "abc123"
   → Returns: {"status": "success", "result": {"output_directory": "...", "files_created": [...]}}

5. Analyze: Use submit_comprehensive_analysis with prediction_results_dir from step 4
   → Returns: New job_id for analysis job
```

### Background Processing Workflow
```
1. Submit multiple jobs in parallel:
   - Use submit_batch_rna_prediction for dataset A
   - Use submit_batch_rna_prediction for dataset B
   - Use submit_large_sequence_prediction for special sequence

2. Monitor all jobs: Use list_jobs
   → Returns: List of all submitted jobs with their status

3. Retrieve completed results as they finish:
   - Use get_job_result for each completed job_id

4. Analyze all results together:
   - Use submit_comprehensive_analysis for combined analysis
```

---

## Error Handling

### Common Error Responses

#### File Not Found
```json
{
  "status": "error",
  "error": "File not found: input.fasta"
}
```

#### Invalid RNA Sequence
```json
{
  "status": "error",
  "error": "Invalid nucleotides found: {'X', 'Y'}. RNA sequences should contain only A, U, G, C."
}
```

#### Job Not Found
```json
{
  "status": "error",
  "error": "Job abc123 not found"
}
```

#### Environment Issues
```json
{
  "status": "error",
  "error": "RhoFold modules not available: No module named 'rhofold'. Ensure you're using the env_py37 environment."
}
```

### Troubleshooting

1. **Import Errors**: Ensure using correct conda environment
   - RhoFold tools require `env_py37` (Python 3.7)
   - Analysis tools can use `env` (Python 3.10)

2. **Job Failures**: Check logs with `get_job_log`
   - Look for specific error messages
   - Verify input file paths and formats

3. **Long Execution Times**:
   - Use GPU (`device: "cuda:0"`) for 5-10x speedup
   - Consider reducing `relax_steps` for faster results

4. **Memory Issues**:
   - Monitor system RAM for very large sequences
   - Use batch processing for multiple sequences

---

## Performance Guidelines

### Execution Time Estimates

| Operation | CPU Runtime | GPU Runtime | Memory Usage |
|-----------|-------------|-------------|--------------|
| Single sequence (88 nt) | ~5 minutes | ~30 seconds | ~4-6 GB |
| Batch (3 sequences) | ~15 minutes | ~2 minutes | ~4-6 GB |
| Structure analysis | ~10 seconds | N/A | ~100 MB |

### Optimization Tips

1. **GPU Usage**: Add `device: "cuda:0"` for significant speedup
2. **Batch Processing**: More efficient than multiple single predictions
3. **Relaxation Steps**: Reduce from 1000 to 100 for faster results (slight quality loss)
4. **Memory Management**: Process large batches in smaller chunks

### Resource Planning

- **Storage**: Expect ~4 MB per predicted structure
- **CPU**: High usage during model inference (25-30 seconds per sequence)
- **Memory**: Peak usage ~4-6 GB during prediction
- **Network**: 508 MB model download on first use

This MCP server provides comprehensive RNA structure prediction capabilities with both immediate and background processing options, suitable for research and production use.