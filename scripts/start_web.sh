#!/bin/bash
# Start the unified web interface for MoE_RAG

echo "Starting MoE_RAG Web Interface on port 8050..."
echo "----------------------------------------"

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "Running in Docker container"
    cd /workspace
    python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload
else
    echo "Running on host system"
    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload
fi