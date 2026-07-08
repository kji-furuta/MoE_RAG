# Suggested Development Commands

## Docker Environment Management

### Start Complete Environment
```bash
# Build and start with RAG dependencies
./scripts/docker_build_rag.sh --no-cache

# Quick start (if already built)
cd docker && docker-compose up -d
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh
```

### Direct Server Start (Debugging)
```bash
docker exec ai-ft-container python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload
```

### Stop Environment
```bash
cd docker && docker-compose down
```

## Testing Commands

### Integration Tests
```bash
python scripts/test_integration.py
python scripts/test_docker_rag.py
python scripts/test_continual_learning_integration.py
python scripts/test_moe_rag_integration.py
```

### Configuration Tests
```bash
python scripts/test_config_resolution.py
python scripts/test_model_path_resolution.py
```

### Feature Tests
```bash
python scripts/simple_feature_test.py
python scripts/test_specialized_features.py
```

## RAG Operations

### Index Documents
```bash
python scripts/rag/index_documents.py
```

### Test RAG Query
```bash
curl -X POST "http://localhost:8050/rag/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "設計速度80km/hの道路の最小曲線半径は？", "top_k": 5}'
```

### Health Check
```bash
curl "http://localhost:8050/rag/health"
```

## Model Training

### Train Large Models
```bash
python scripts/train_large_model.py
python scripts/train_calm3_22b.py
```

### LoRA Fine-tuning
```bash
python scripts/test/simple_lora_tutorial.py
```

### Continual Learning
```bash
python src/training/continual_learning_pipeline.py
```

## Ollama Operations

### Start Ollama Service
```bash
ollama serve
```

### Pull Model
```bash
ollama pull llama3.2:3b
```

### List Models
```bash
ollama list
```

## Code Quality

### Format Code
```bash
black src/ app/ scripts/ --line-length 88
```

### Sort Imports
```bash
isort src/ app/ scripts/ --profile black
```

### Run Linter
```bash
flake8 src/ app/ scripts/
```

### Run Tests
```bash
pytest tests/
```

## Git Operations

### Status Check
```bash
git status
```

### Push to Repository
```bash
git add .
git commit -m "Your commit message"
git push moe_rag main
```

## Monitoring

### Check Logs
```bash
docker logs -f ai-ft-container --tail 50
```

### GPU Status
```bash
nvidia-smi
```

### System Resources
```bash
docker exec ai-ft-container python scripts/system_status_report.py
```

## Useful Docker Commands

### Enter Container
```bash
docker exec -it ai-ft-container bash
```

### Check Running Containers
```bash
docker ps
```

### Clean Docker Resources
```bash
docker system prune -a
docker builder prune
```