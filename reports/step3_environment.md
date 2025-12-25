# Step 3: Environment Setup Report

## Python Version Detection
- **Detected Python Version**: 3.7.12
- **Strategy**: Dual environment setup (Main MCP + Legacy RhoFold)

## Main MCP Environment
- **Location**: ./env
- **Python Version**: 3.10.19 (for MCP server compatibility)
- **Package Manager**: mamba
- **Creation Command**: `mamba create -p ./env python=3.10 pip -y`

## Legacy Build Environment
- **Location**: ./env_py37
- **Python Version**: 3.7.12 (original detected version)
- **Purpose**: RhoFold dependencies requiring specific Python 3.7
- **Creation Command**: `mamba env create -f repo/RhoFold/envs/environment_linux.yaml -p ./env_py37`

## Dependencies Installed

### Main Environment (./env)
Core MCP dependencies:
- fastmcp==2.14.1 (with all sub-dependencies)
- loguru==0.7.3
- click==8.3.1
- pandas==2.3.3
- numpy==2.2.6
- tqdm==4.67.1

FastMCP sub-dependencies include:
- mcp==1.25.0
- pydantic==2.12.5
- uvicorn==0.40.0
- websockets==15.0.1
- And 50+ other packages for full MCP functionality

### Legacy Environment (./env_py37)
RhoFold and scientific computing dependencies:
- rhofold==0.0.1 (installed via `pip install -e .`)
- pytorch==1.10.2 (with CUDA 11.3 support)
- cudatoolkit==11.3.1
- biopython==1.79
- transformers==4.29.2
- openmm==7.7.0
- numpy==1.21.2
- scipy==1.7.1
- matplotlib==3.5.3
- pandas==1.3.5
- huggingface_hub==0.10.1

And 100+ other scientific packages from environment_linux.yaml

## Activation Commands
```bash
# Main MCP environment
mamba run -p ./env python script.py

# Legacy environment (for RhoFold scripts)
mamba run -p ./env_py37 python script.py
```

## Verification Status
- [x] Main environment (./env) functional - FastMCP 2.14.1 imported successfully
- [x] Legacy environment (./env_py37) functional - RhoFold imported successfully
- [x] Core imports working in both environments
- [x] RhoFold package installation successful
- [x] Package manager: mamba (preferred over conda for speed)

## Installation Timeline
1. **Package Manager Check**: mamba detected and selected
2. **Main Environment**: Created in ~30 seconds
3. **MCP Dependencies**: Installed in ~60 seconds
4. **Legacy Environment**: Created in ~4 minutes (large packages: CUDA toolkit, PyTorch)
5. **RhoFold Package**: Installed in ~30 seconds
6. **Total Setup Time**: ~6 minutes

## Environment Sizes
- **Main Environment**: ~1.2GB (primarily FastMCP and dependencies)
- **Legacy Environment**: ~8.5GB (primarily CUDA toolkit 856MB + PyTorch 1GB + scientific libs)
- **Total Disk Usage**: ~9.7GB

## Notes
- **Dual Environment Rationale**: FastMCP requires Python 3.10+, but RhoFold environment is locked to Python 3.7.12 due to specific dependency versions (PyTorch 1.10.2, CUDA 11.3, etc.)
- **CUDA Support**: Legacy environment includes CUDA 11.3.1 toolkit for GPU acceleration
- **Package Conflicts**: No conflicts detected between environments due to isolation
- **Memory Usage**: Peak RAM usage during installation was ~4GB (CUDA toolkit extraction)

## Recommended Usage Pattern
1. Use **./env** for MCP server and analysis scripts (use_case_4)
2. Use **./env_py37** for RhoFold prediction scripts (use_cases 1-3)
3. Both environments are completely isolated and can coexist safely
4. Switch environments using `mamba run -p ./env` or `mamba run -p ./env_py37` commands