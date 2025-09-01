# Architecture Overview

## System Architecture

### 1. Web Layer (Port 8050)
- **FastAPI Application** (`app/main_unified.py`)
  - Serves both REST APIs and WebSocket connections
  - Static file serving for web assets
  - Template rendering with Jinja2
  - CORS configuration for localhost access

### 2. Core Systems

#### Fine-tuning System (`src/training/`)
- **LoRA**: Parameter-efficient fine-tuning with rank adaptation
- **DoRA**: Weight-decomposed LoRA for improved accuracy
- **Full Fine-tuning**: Complete model parameter updates
- **Continual Learning**: EWC-based catastrophic forgetting prevention
- **Multi-GPU**: DeepSpeed integration for distributed training

#### RAG System (`src/rag/`)
- **Document Processing Pipeline**:
  1. PDF upload → OCR (EasyOCR) → Text extraction
  2. Chunking (512 tokens, 128 overlap)
  3. Embedding (multilingual-e5-large)
  4. Vector storage (Qdrant)
  
- **Query Processing**:
  1. Query embedding generation
  2. Hybrid search (vector 0.7 + keyword 0.3)
  3. Re-ranking with Cross-encoder
  4. Response generation with citation

#### Inference System (`src/inference/`)
- **vLLM**: PagedAttention for efficient KV cache management
- **AWQ Quantization**: 4-bit quantization for 75% memory reduction
- **Ollama Integration**: Local model deployment with API

### 3. Data Flow

```
User Request → FastAPI Router → Service Layer → Core Logic
                                      ↓
                            Background Task Executor
                                      ↓
                            Model/Database Operations
                                      ↓
                               Response Generation
```

### 4. Storage Architecture

#### File System
```
/workspace/
├── data/               # Training data, uploaded files
├── outputs/            # Trained models, checkpoints
├── temp_uploads/       # Temporary file storage
├── qdrant_data/       # Vector database storage
├── metadata/          # SQLite databases
└── logs/             # Application and training logs
```

#### Databases
- **Qdrant**: Vector similarity search (port 6333)
- **SQLite**: Document metadata, search history
- **Redis** (optional): Caching layer

### 5. Model Management

#### Model Loading Strategy
- Dynamic quantization based on model size:
  - 32B/22B models → 4-bit quantization
  - 7B/8B models → 8-bit quantization
  - <7B models → FP16/BF16
- CPU offloading for large models
- Model caching to reduce loading time

#### Model Storage
- Base models: HuggingFace cache
- Fine-tuned: `/workspace/outputs/`
- LoRA adapters: `/workspace/outputs/lora_*`
- Continual learning: `/workspace/outputs/continual_task_*`
- MoE models: `/workspace/outputs/moe_*`

### 6. Container Architecture

```yaml
Services:
- ai-ft-container:     # Main application
  - Ports: 8050 (web), 8888 (jupyter), 6006 (tensorboard)
  - GPU access enabled
  - 32GB shared memory
  
- ai-ft-qdrant:        # Vector database
  - Port: 6333 (HTTP), 6334 (gRPC)
  - Persistent volume for data
  
- Ollama (integrated): # Local LLM service
  - Port: 11434
  - Model: Llama 3.2 3B
```

### 7. Security & Performance

#### Security
- Environment variables for sensitive tokens
- CORS restricted to localhost
- File upload validation
- Input sanitization

#### Performance Optimizations
- Async/await for I/O operations
- Background task processing
- Connection pooling
- Batch processing support
- GPU memory management
- Model quantization

### 8. Monitoring & Logging

- **Application logs**: Python logging to files and console
- **Training metrics**: TensorBoard, Weights & Biases
- **System metrics**: GPU usage, memory monitoring
- **Request tracking**: FastAPI middleware logging