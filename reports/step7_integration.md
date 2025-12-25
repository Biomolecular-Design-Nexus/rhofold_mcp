# RhoFold MCP Integration Test Results

## Test Information
- **Test Date**: 2025-12-24T11:41:54.133355
- **Server Name**: RhoFold
- **Server Path**: src/server.py

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | 6 |
| Passed | 4 |
| Failed | 2 |
| Pass Rate | 66.7% |
| Ready for Production | ❌ No |

## Detailed Test Results

### server_startup
- **Status**: ✅ Passed
- **Output**: Server imports and initializes correctly

### tool_availability
- **Status**: ✅ Passed
- **Expected**: 14
- **Found**: 14

### examples_availability
- **Status**: ✅ Passed
- **Example_Dirs**: 16
- **Fasta_Files**: 3

### job_system
- **Status**: ✅ Passed
- **Jobs_Dir**: /home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/rhofold_mcp/jobs
- **Existing_Jobs**: 0

### sync_tool_validate_rna_sequence
- **Status**: ⚠️ Error
- **Error**: 'FunctionTool' object is not callable

### sync_tool_get_example_data_info
- **Status**: ⚠️ Error
- **Error**: 'FunctionTool' object is not callable

