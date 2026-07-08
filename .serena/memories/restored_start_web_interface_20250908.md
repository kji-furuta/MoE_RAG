# Restored start_web_interface.sh - 2025-09-08

## Purpose
Restored the deleted `start_web_interface.sh` script that provides essential initialization functionality for the MoE_RAG web interface.

## Key Features Restored
1. **Ollama Service Management**
   - Automatically starts Ollama service
   - Checks and downloads llama3.2:3b model if needed
   - Port 11434

2. **Permission Management**
   - Updated to use `/workspace/scripts/setup/fix_permissions.sh` (since setup_permissions.sh was deleted)
   - Fallback directory creation if script not found
   - Sets proper permissions for critical directories

3. **Continual Learning Initialization**
   - Creates `continual_learning_config.yaml` if not exists
   - Configures base models and EWC settings
   - Sets training parameters

4. **Web Server Launch**
   - Shows all available endpoints
   - Starts uvicorn on port 8050 with reload

## Updates Made
- Changed from `setup_permissions.sh` to `fix_permissions.sh` (reflecting cleanup changes)
- Added existence check for config file to avoid overwriting
- Added fallback chmod commands for essential directories

## Usage
```bash
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh
```