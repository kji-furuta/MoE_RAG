# MoE-RAG System Architecture

## Overview

MoE-RAG is a unified AI fine-tuning and RAG platform specialized for Japanese civil engineering and road design. The system integrates three core capabilities:

1. **Fine-tuning Engine** - LoRA/DoRA/full parameter training
2. **RAG System** - Hybrid vector + keyword search for technical documents
3. **Continual Learning** - EWC-based multi-task learning without catastrophic forgetting

All services are unified under a single FastAPI server on port 8050, running in a containerized environment with GPU acceleration.

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   Web Browser (Client)                        │
└────────────────────────┬─────────────────────────────────────┘
                         │ HTTP/WebSocket
┌────────────────────────▼─────────────────────────────────────┐
│              FastAPI Server (Port 8050)                       │
│                 app/main_unified.py                           │
├───────────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌──────────┐  ┌──────────────────┐         │
│  │ Fine-tune  │  │   RAG    │  │ Continual Learn  │         │
│  │   Routes   │  │  Routes  │  │     Routes       │         │
│  └──────┬─────┘  └────┬─────┘  └────────┬─────────┘         │
└─────────┼──────────────┼─────────────────┼───────────────────┘
          │              │                 │
┌─────────▼──────────────▼─────────────────▼───────────────────┐
│                  Core Services Layer                          │
├───────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌────────────┐  ┌──────────────┐          │
│  │  Training   │  │    RAG     │  │     MoE      │          │
│  │   Engine    │  │   Engine   │  │   Manager    │          │
│  └─────────────┘  └────────────┘  └──────────────┘          │
└───────────────────────────────────────────────────────────────┘
          │              │                 │
┌─────────▼──────────────▼─────────────────▼───────────────────┐
│              Infrastructure Layer                             │
├───────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌────────────┐  ┌──────────────┐          │
│  │   Models    │  │   Qdrant   │  │    Ollama    │          │
│  │  Storage    │  │   Vector   │  │   Service    │          │
│  │             │  │     DB     │  │ (Port 11434) │          │
│  └─────────────┘  └────────────┘  └──────────────┘          │
└───────────────────────────────────────────────────────────────┘
```

## Architectural Layers

### 1. Presentation Layer

**Location**: `templates/`, `app/static/`

HTML templates using Jinja2 with Bootstrap:
- `base.html` - Base layout with navigation
- `index.html` - Dashboard
- `finetune.html` - Fine-tuning interface
- `rag.html` - RAG search interface
- `continual_learning.html` - Continual learning management

Static resources:
- Bootstrap CSS/JS
- Custom JavaScript for UI interactions
- Logo and image assets (Teikoku.co branding)

### 2. API Routing Layer

**Location**: `app/main_unified.py`, `app/routers/`

Main router with endpoint groups:

| Prefix | Purpose | Key Endpoints |
|--------|---------|---------------|
| `/api/` | Fine-tuning | `/train`, `/training-status/{id}`, `/generate` |
| `/rag/` | RAG search | `/query`, `/upload-document`, `/documents` |
| `/api/continual/` | Continual learning | `/train`, `/tasks`, `/task/{id}` |
| `/api/moe/` | MoE management | `/experts`, `/routing-stats` |

### 3. Business Logic Layer

**Location**: `src/`

<details>
<summary><b>Fine-tuning Services</b> (<code>src/training/</code>)</summary>

- **LoRATrainer** - Parameter-efficient fine-tuning with low-rank adaptation
- **DoRAImplementation** - Weight-decomposed LoRA for improved accuracy
- **EWCContinualLearner** - Elastic Weight Consolidation for multi-task learning
- **MultiGPUTrainingManager** - Distributed training across multiple GPUs

Key features:
- Dynamic quantization (4-bit/8-bit based on model size)
- CPU offloading for large models
- Gradient checkpointing for memory efficiency
- Mixed precision training (bf16/fp16)
</details>

<details>
<summary><b>RAG Services</b> (<code>src/rag/</code>)</summary>

- **QueryEngine** - Main query processing pipeline
- **VectorStore** - Qdrant-based vector storage and retrieval
- **HybridSearch** - Combined vector (0.7) + keyword (0.3) search
- **DocumentProcessor** - PDF parsing, OCR, table extraction

Key features:
- Multilingual-e5-large embeddings (1024 dimensions)
- Semantic chunking (512 tokens, 128 overlap)
- BM25 keyword search
- Specialized road design validation
</details>

<details>
<summary><b>Inference Services</b> (<code>src/inference/</code>)</summary>

- **VLLMIntegration** - PagedAttention-based high-speed inference
- **AWQQuantizer** - 4-bit quantization (75% memory reduction)
- **ModelLoader** - Dynamic model loading with memory optimization

Key features:
- Ollama integration for Llama models
- Dynamic batch sizing
- KV cache management
- Multi-GPU inference
</details>

### 4. Data Access Layer

<details>
<summary><b>Vector Store</b></summary>

**Qdrant** (Port 6333):
- Connection: `src/rag/indexing/vector_store.py`
- Schema: UUID-based points with metadata
- Collections: `road_design_docs`
- Dimensions: 1024 (multilingual-e5-large)
</details>

<details>
<summary><b>Model Storage</b></summary>

**Local Storage** (`outputs/`):
- LoRA adapters (safetensors format)
- Full fine-tuned models
- EWC data (Fisher matrices, task history)
- Continual learning checkpoints

**Ollama** (Port 11434):
- GGUF format models
- Dynamic registration via `scripts/init_ollama_models.sh`
- Automatic model detection with wildcards
</details>

<details>
<summary><b>Metadata Management</b></summary>

- **Task State**: `data/continual_learning/tasks_state.json`
- **EWC Data**: `outputs/ewc_data/`
- **Search History**: SQLite/JSON-based storage
</details>

### 5. Infrastructure Layer

**Location**: `docker/`

Docker services:
```yaml
ai-ft-container:
  - FastAPI server (port 8050)
  - All integrated services
  - GPU support (NVIDIA Container Toolkit)

qdrant:
  - Vector database
  - Port 6333

ollama:
  - LLM service
  - Port 11434
```

Resource management:
- GPU allocation via CUDA_VISIBLE_DEVICES
- Dynamic quantization based on available memory
- CPU offloading for large models
- Process management via Uvicorn workers

## Data Flow Patterns

### Fine-tuning Flow

```
User Request → API Validation → Task Queue
     ↓
Training Manager → Model Loader → GPU Training
     ↓                               ↓
Progress Updates → WebSocket ← Training Metrics
     ↓
Task Completion → Model Save → Ollama Registration
```

Key steps:
1. Request validation (Pydantic models)
2. Background task creation (FastAPI BackgroundTasks)
3. Model loading with quantization
4. LoRA/DoRA training loop
5. Checkpoint saving (every 500 steps)
6. Final model export (safetensors)
7. Optional GGUF conversion and Ollama registration

### RAG Query Flow

```
Query → Embedding → Vector Search ──┐
  ↓         ↓            ↓           │
  │     E5-Large    Qdrant Top-K    │
  │                                  ├→ Hybrid Ranking
  └→ BM25 Search → Keyword Match ───┘        ↓
                                    Retrieved Docs
                                         ↓
                        LLM Generation ← Context Assembly
                                         ↓
                                      Response
```

Key steps:
1. Query embedding (multilingual-e5-large)
2. Parallel vector + keyword search
3. Score fusion (0.7 vector + 0.3 keyword)
4. Reranking with cross-encoder
5. Context assembly with citations
6. LLM generation (Ollama/vLLM)
7. Response formatting

### Continual Learning Flow

```
New Task → EWC Calculator → Fisher Matrix
    ↓           ↓                ↓
Task Data  Important Weights  Save to Disk
    ↓                              ↓
Training Loop → Regularization Loss
    ↓                ↓
Standard Loss + λ * EWC Loss
    ↓
Model Update → Save Checkpoint → Update Task State
```

Key steps:
1. Fisher Information Matrix calculation (previous task)
2. New task data loading
3. Training with EWC regularization (λ=5000)
4. Checkpoint saving
5. Task state update (`tasks_state.json`)
6. Optional model merging

## Configuration Management

### Environment Variables

```bash
# Authentication
HF_TOKEN              # HuggingFace Hub access
WANDB_API_KEY         # Weights & Biases logging

# Hardware
CUDA_VISIBLE_DEVICES  # GPU selection
MODEL_CACHE_DIR       # Model cache location

# Services
QDRANT_URL           # Vector DB endpoint
OLLAMA_BASE_URL      # Ollama service
```

### Configuration Files

| File | Purpose |
|------|---------|
| `config/model_config.yaml` | Model definitions, quantization settings |
| `src/rag/config/rag_config.yaml` | RAG parameters, search weights |
| `configs/` | Training templates (LoRA, DoRA, full) |
| `docker/docker-compose.yml` | Service orchestration |

## Security Architecture

### Authentication & Authorization
- **Current**: Basic authentication (development)
- **Planned**: JWT-based token auth
- **API Keys**: Environment variable management
- **CORS**: Restricted to localhost:8050

### Data Protection
- **Input Validation**: Pydantic models
- **File Upload Limits**: 200MB max
- **SQL Injection**: Parameterized queries
- **Path Traversal**: Sanitized file paths

## Scalability Considerations

### Horizontal Scaling
- Multi-worker support (Uvicorn/Gunicorn)
- Stateless API design (task state in JSON/DB)
- Load balancer ready
- Distributed caching (Redis-ready)

### Vertical Scaling
- Multi-GPU training (DDP, FSDP)
- Memory optimization (quantization, offloading)
- Batch processing optimization
- Async I/O for non-blocking operations

## Monitoring & Observability

### Logging
- Python standard logging (`logging` module)
- Log rotation (daily, 7-day retention)
- Structured logs (JSON format)
- Error tracking with stack traces

### Metrics
- `/metrics` endpoint (Prometheus-compatible planned)
- GPU utilization (nvidia-smi)
- Response time tracking
- Training progress metrics

### Health Checks
- `/health` endpoint (liveness)
- `/rag/health` endpoint (RAG system)
- Service dependency checks (Qdrant, Ollama)

## Deployment Patterns

### Development

```bash
# Build and start
./scripts/docker_build_rag.sh --no-cache

# Run with hot reload
docker exec ai-ft-container python -m uvicorn app.main_unified:app \
  --host 0.0.0.0 --port 8050 --reload
```

### Production

```bash
# Docker Compose
docker-compose up -d

# With Gunicorn (multiple workers)
gunicorn app.main_unified:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8050
```

## Future Architecture Evolution

### Phase 1: Microservices
- Separate fine-tuning, RAG, and continual learning services
- gRPC for inter-service communication
- Service mesh (Istio/Linkerd)

### Phase 2: Kubernetes
- Helm charts for deployment
- Horizontal pod autoscaling
- StatefulSets for stateful services

### Phase 3: API Gateway
- Unified entry point (Kong/Ambassador)
- Rate limiting and throttling
- JWT validation at gateway

### Phase 4: Event-Driven
- Message queue (RabbitMQ/Kafka)
- Async task processing (Celery)
- Event sourcing for training history

### Phase 5: Distributed Infrastructure
- Redis for distributed caching
- PostgreSQL for metadata
- MinIO for model artifacts
- Elasticsearch for log aggregation

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Qdrant Vector Database](https://qdrant.tech/)
- [PEFT (LoRA/DoRA)](https://github.com/huggingface/peft)
- [vLLM Inference Engine](https://github.com/vllm-project/vllm)
- [Ollama](https://ollama.ai/)