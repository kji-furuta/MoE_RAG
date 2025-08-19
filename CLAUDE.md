# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Fine-tuning Toolkit (AI_FT_7) is a comprehensive Japanese LLM fine-tuning platform with integrated RAG (Retrieval-Augmented Generation) system specialized for civil engineering and road design. The project provides a unified web interface for both fine-tuning and RAG functionality through a single port (8050).

## Key Architecture Components

### 1. Unified Web Interface
- **Main Server**: `app/main_unified.py` - FastAPI application serving both fine-tuning and RAG APIs on port 8050
- **Static Files**: Located in `app/static/` and `static/` directories
- **Templates**: HTML templates in `templates/` directory with Bootstrap integration
- **Logo Assets**: Teikoku University logo files (`logo_teikoku.png/jpg`) in multiple locations

### 2. Core Systems

#### Fine-tuning System (`src/training/`)
- **LoRA Training**: `lora_finetuning.py` - Parameter-efficient fine-tuning
- **Full Fine-tuning**: `full_finetuning.py` - Complete model training
- **DoRA**: `dora/dora_implementation.py` - Weight-decomposed LoRA for improved accuracy
- **EWC**: `ewc_utils.py` - Elastic Weight Consolidation for continual learning
- **Multi-GPU**: `multi_gpu_training.py` - Distributed training support
- **Continual Learning**: `continual_learning_pipeline.py` - Task-based continuous learning

#### RAG System (`src/rag/`)
- **Query Engine**: `core/query_engine.py` - Main RAG processing engine
- **Vector Store**: `indexing/vector_store.py` - Qdrant-based vector storage
- **Hybrid Search**: `retrieval/hybrid_search.py` - Combined vector + keyword search
- **Document Processing**: `document_processing/` - PDF, OCR, and table extraction
- **Specialized Features**: `specialized/` - Numerical processing, design standard validation

#### Inference System (`src/inference/`)
- **vLLM Integration**: `vllm_integration.py` - PagedAttention-based high-speed inference
- **AWQ Quantization**: `awq_quantization.py` - 4-bit quantization for memory reduction

### 3. Model Management
- **Model Loader**: `app/memory_optimized_loader.py` - Dynamic quantization and memory optimization
- **Supported Models**: Located in `outputs/` directory, including LoRA adapters and full models
- **Model Config**: `config/model_config.yaml` and `configs/` directory for model specifications
- **Ollama Integration**: Support for Llama 3.2 3B via Ollama API (port 11434)

## Common Development Commands

### Docker Operations
```bash
# Build and start the complete environment (recommended)
./scripts/docker_build_rag.sh --no-cache

# Manual Docker operations
cd docker
docker-compose up -d --build  # First time
docker-compose up -d           # Subsequent starts

# Start web interface
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh

# Direct server start (for debugging)
docker exec ai-ft-container python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload
```

### Testing
```bash
# Integration tests
python scripts/test_integration.py
python scripts/test_docker_rag.py
python scripts/test_continual_learning_integration.py

# Configuration tests
python scripts/test_config_resolution.py
python scripts/test_model_path_resolution.py

# Feature tests
python scripts/simple_feature_test.py
python scripts/test_specialized_features.py
```

### RAG Operations
```bash
# Index documents
python scripts/rag/index_documents.py

# Test RAG queries
curl -X POST "http://localhost:8050/rag/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "設計速度80km/hの道路の最小曲線半径は？", "top_k": 5}'

# Health check
curl "http://localhost:8050/rag/health"
```

### Model Training
```bash
# Train large models
python scripts/train_large_model.py
python scripts/train_calm3_22b.py

# Run LoRA fine-tuning
python scripts/test/simple_lora_tutorial.py

# Continual learning
python src/training/continual_learning_pipeline.py
```

### Ollama Operations
```bash
# Start Ollama service
ollama serve

# Pull Llama model
ollama pull llama3.2:3b

# List available models
ollama list
```

## API Endpoints

### Fine-tuning APIs
- `POST /api/train` - Start model training
- `GET /api/training-status/{task_id}` - Check training status
- `POST /api/generate` - Generate text
- `GET /api/models` - List available models

### RAG APIs
- `GET /rag/health` - System health check
- `POST /rag/query` - Advanced document search
- `POST /rag/upload-document` - Upload documents
- `GET /rag/documents` - List documents
- `POST /rag/stream-query` - Streaming search
- `GET /rag/system-info` - System information

### Continual Learning APIs
- `GET /continual` - Continual learning interface
- `POST /api/continual/train` - Start continual learning task
- `GET /api/continual/tasks` - List all tasks
- `GET /api/continual/task/{task_id}` - Get task status
- `POST /api/continual/update-models` - Refresh available models list

### Web Interface Routes
- `/` - Main dashboard
- `/finetune` - Fine-tuning interface
- `/rag` - RAG interface
- `/continual` - Continual learning interface
- `/models` - Model management
- `/manual` - User manual
- `/system-overview` - System documentation

## Critical Implementation Details

### Memory Optimization
- The system uses dynamic quantization based on model size:
  - 32B/22B models: 4-bit quantization
  - 7B/8B models: 8-bit quantization
- CPU offloading is enabled for large models
- Model caching is implemented to reduce loading times
- AWQ quantization available for 75% memory reduction

### RAG Vector Store
- Uses Qdrant with UUID-based point IDs
- Hybrid search combines vector similarity (0.7 weight) and keyword matching (0.3 weight)
- Supports multilingual-e5-large embeddings
- Document chunks are 512 tokens with 128 token overlap

### Continual Learning System
- EWC-based learning with Fisher Information Matrix
- Default EWC lambda: 5000
- Task history stored in `outputs/ewc_data/task_history.json`
- Fisher matrices saved as `outputs/ewc_data/fisher_task_*.pt`
- Models saved to `outputs/continual_task_*`
- Tasks state tracked in `data/continual_learning/tasks_state.json`

### Error Handling Patterns
- Bootstrap JavaScript errors are handled in `templates/base.html`
- RAG search history modal requires proper Bootstrap initialization
- File uploads use multipart form data with progress tracking
- Training tasks run asynchronously with background task management

### Security Considerations
- CORS is restricted to localhost:8050 and 127.0.0.1:8050
- File uploads are validated for type and size
- Sensitive tokens (HF_TOKEN, WANDB_API_KEY) are managed via environment variables
- Docker container runs with GPU capabilities enabled

## Troubleshooting Common Issues

### Web Interface Not Starting
1. Check if port 8050 is already in use: `netstat -tlnp | grep 8050`
2. Verify files are mounted correctly: `docker exec ai-ft-container ls -la /workspace/app/`
3. Check logs: `docker logs ai-ft-container --tail 50`

### RAG System Errors
1. Verify Qdrant is accessible
2. Check embedding model is loaded: `ls ~/.cache/huggingface/`
3. Test vector store connection: `python scripts/test_docker_rag.py`

### GPU/Memory Issues
1. Monitor GPU usage: `nvidia-smi`
2. Check available memory: `free -h`
3. Reduce batch size in training config
4. Enable CPU offloading for large models

### Module Import Errors
1. Verify Python path: `docker exec ai-ft-container python -c "import sys; print(sys.path)"`
2. Check module installation: `docker exec ai-ft-container pip list | grep <module>`
3. Reinstall requirements if needed: `docker exec ai-ft-container pip install -r requirements.txt`

### Ollama Connection Issues
1. Verify Ollama is running: `curl http://localhost:11434/api/tags`
2. Check if model is downloaded: `ollama list`
3. Pull model if missing: `ollama pull llama3.2:3b`
4. Restart Ollama service: `killall ollama && ollama serve`

## Development Workflow Tips

1. **Always use the Docker environment** for consistency
2. **Test changes locally** before deploying using the test scripts
3. **Monitor logs** during development: `docker logs -f ai-ft-container`
4. **Use the unified API** at port 8050 rather than separate services
5. **Check GPU memory** before training large models: `nvidia-smi`
6. **Backup trained models** from the `outputs/` directory regularly
7. **Review task history** in `data/continual_learning/tasks_state.json` for continual learning
8. **Keep Ollama running** in a separate terminal for Ollama model integration