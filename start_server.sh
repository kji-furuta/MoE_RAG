#!/bin/bash

# Activate virtual environment and start server
cd /home/kjifuruta/AI_FT/AI_FT_3
source ai_ft_env/bin/activate

echo "Starting AI FT server with Ollama fallback support..."
echo "Server will be available at: http://localhost:8050"

# Start the server
python3 -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload