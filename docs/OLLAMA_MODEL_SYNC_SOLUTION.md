# Ollama Model Synchronization Solution

## Problem Summary
Ollama models in the Docker container were not automatically appearing in the RAG system's model list after container restart. The startup script's Ollama model update functionality was not working due to timing issues.

## Root Cause
1. **Timing Issue**: The `update_ollama_models_config.py` script was executing before Ollama service was fully initialized
2. **Service Readiness**: Even when Ollama process was running, it wasn't immediately ready to respond to API calls
3. **Synchronous Execution**: The startup script was trying to update models synchronously, blocking the main startup process

## Solution Components

### 1. Enhanced Update Script (`scripts/update_ollama_models_config.py`)
- Added `wait_for_ollama_ready()` function with 30-second timeout
- Checks both process existence and actual API responsiveness
- Updates both RAG config and creates model registry

### 2. Wait and Update Script (`scripts/wait_and_update_ollama.sh`)
- Dedicated script for waiting and updating Ollama models
- Runs independently with proper error handling
- Ensures Ollama is fully ready before attempting update

### 3. Service Initialization Script (`scripts/init_services.sh`)
- Orchestrates all services in proper order:
  1. Start Ollama service and wait for readiness
  2. Initialize GGUF model registry
  3. Update Ollama model configuration
  4. Check and fix permissions
- Can be run manually for troubleshooting

### 4. Modified Startup Script (`scripts/start_web_interface.sh`)
- Runs Ollama update in background to avoid blocking
- Uses nohup to detach the update process
- Logs output to `/workspace/logs/ollama_update.log`

## Files Modified/Created

1. `/scripts/update_ollama_models_config.py` - Enhanced with wait functionality
2. `/scripts/wait_and_update_ollama.sh` - New dedicated wait script
3. `/scripts/init_services.sh` - New comprehensive initialization script
4. `/scripts/start_web_interface.sh` - Modified to run updates asynchronously

## How It Works

1. **Container Start**: When container starts, `start_web_interface.sh` is executed
2. **Ollama Service**: Ollama service starts in background
3. **Background Update**: `wait_and_update_ollama.sh` runs in background
4. **Wait Loop**: Script waits up to 30 seconds for Ollama to be ready
5. **Model Sync**: Once ready, updates RAG config with available Ollama models
6. **Registry Creation**: Creates `/workspace/models/ollama_models.json` registry

## Verification

### Manual Test
```bash
# Run initialization script
docker exec ai-ft-container bash /workspace/scripts/init_services.sh

# Check Ollama models
docker exec ai-ft-container ollama list

# Verify RAG config
docker exec ai-ft-container grep -A 10 "available_models:" /workspace/src/rag/config/rag_config.yaml
```

### Automatic on Restart
```bash
# Restart container
docker restart ai-ft-container

# Start web interface
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh

# Check logs
docker exec ai-ft-container cat /workspace/logs/ollama_update.log
```

## Benefits

1. **Reliability**: Ensures Ollama is ready before attempting updates
2. **Non-blocking**: Main startup process continues while models update
3. **Logging**: All operations are logged for debugging
4. **Fallback**: Multiple fallback mechanisms if primary method fails
5. **Persistence**: Models are properly registered and persist across restarts

## Troubleshooting

If models still don't appear:

1. Check Ollama service status:
   ```bash
   docker exec ai-ft-container pgrep -x ollama
   ```

2. Check update logs:
   ```bash
   docker exec ai-ft-container cat /workspace/logs/ollama_update.log
   ```

3. Run manual update:
   ```bash
   docker exec ai-ft-container python3 /workspace/scripts/update_ollama_models_config.py
   ```

4. Verify model registry:
   ```bash
   docker exec ai-ft-container cat /workspace/models/ollama_models.json
   ```

## Future Improvements

1. Add health check endpoint for Ollama readiness
2. Implement retry mechanism with exponential backoff
3. Add webhook notification when models are synchronized
4. Create systemd-style service management for better control