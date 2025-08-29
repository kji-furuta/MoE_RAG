# MoE_RAG Project Structure Documentation

## Overview

MoE_RAG is a comprehensive Japanese LLM fine-tuning platform with integrated RAG (Retrieval-Augmented Generation) system specialized for civil engineering and road design. The project provides a unified web interface serving both fine-tuning and RAG functionality through a single FastAPI application on port 8050.

**Technology Stack:**
- Backend: FastAPI, Python
- AI/ML: Transformers, LoRA, DoRA, vLLM, AWQ Quantization
- Vector Database: Qdrant
- Containerization: Docker, Docker Compose
- Monitoring: Prometheus, Grafana
- Model Integration: Ollama, HuggingFace

## Directory Tree

```
MoE_RAG/
├── app/                                    # Main FastAPI application
│   ├── main_unified.py                     # Unified FastAPI server (port 8050)
│   ├── model_utils.py                      # Model loading and utilities
│   ├── memory_optimized_loader.py          # Dynamic quantization and memory optimization
│   ├── ollama_integration.py               # Ollama API integration
│   ├── routers/                            # FastAPI route modules
│   │   ├── finetuning.py                   # Fine-tuning endpoints
│   │   ├── rag.py                          # RAG system endpoints
│   │   ├── continual.py                    # Continual learning endpoints
│   │   └── models.py                       # Model management endpoints
│   ├── static/                             # Web UI static files
│   │   ├── moe_rag_ui.html                 # Main interface
│   │   ├── moe_training.html               # Training interface
│   │   ├── logo_teikoku.png/jpg            # Teikoku.co branding
│   │   └── continual_learning/             # Continual learning UI
│   ├── monitoring/                         # System monitoring
│   └── continual_learning/                 # Continual learning management
│       ├── continual_learning_ui.py        # UI components
│       └── task_scheduler.py               # Task scheduling
├── src/                                    # Core source code modules
│   ├── training/                           # Model training systems
│   │   ├── lora_finetuning.py              # LoRA parameter-efficient training
│   │   ├── full_finetuning.py              # Complete model training
│   │   ├── continual_learning_pipeline.py  # Continual learning system
│   │   ├── multi_gpu_training.py           # Distributed training support
│   │   ├── ewc_utils.py                    # Elastic Weight Consolidation
│   │   └── dora/                           # DoRA implementation
│   │       └── dora_implementation.py      # Weight-decomposed LoRA
│   ├── rag/                                # RAG system implementation
│   │   ├── core/                           # Core RAG components
│   │   │   ├── query_engine.py             # Main RAG processing engine
│   │   │   ├── async_query_engine.py       # Async query processing
│   │   │   └── citation_engine.py          # Source citation system
│   │   ├── indexing/                       # Document indexing
│   │   │   ├── vector_store.py             # Qdrant vector storage
│   │   │   ├── embedding_model.py          # Embedding model management
│   │   │   └── metadata_manager.py         # Document metadata handling
│   │   ├── retrieval/                      # Information retrieval
│   │   │   ├── hybrid_search.py            # Vector + keyword search
│   │   │   └── reranker.py                 # Result reranking
│   │   ├── document_processing/            # Document handling
│   │   │   ├── pdf_processor.py            # PDF document processing
│   │   │   ├── ocr_processor.py            # OCR text extraction
│   │   │   ├── table_extractor.py          # Table data extraction
│   │   │   └── chunking_strategy.py        # Document chunking
│   │   ├── specialized/                    # Domain-specific features
│   │   │   ├── numerical_processor.py      # Numerical calculation handling
│   │   │   ├── calculation_validator.py    # Engineering calculation validation
│   │   │   └── version_manager.py          # Document version control
│   │   ├── config/                         # RAG configuration
│   │   │   ├── rag_config.yaml             # Main RAG configuration
│   │   │   ├── prompt_templates.yaml       # Query prompt templates
│   │   │   └── model_path_resolver.py      # Model path resolution
│   │   └── monitoring/                     # RAG system monitoring
│   │       ├── health_check.py             # System health monitoring
│   │       └── metrics.py                  # Performance metrics
│   ├── moe/                                # Mixture of Experts implementation
│   │   ├── moe_architecture.py             # MoE model architecture
│   │   ├── moe_training.py                 # MoE-specific training
│   │   └── data_preparation.py             # MoE data preprocessing
│   ├── moe_rag_integration/                # MoE-RAG integration layer
│   │   ├── unified_moe_rag_system.py       # Unified system implementation
│   │   ├── expert_router.py                # Expert routing logic
│   │   ├── hybrid_query_engine.py          # MoE-RAG query processing
│   │   └── response_fusion.py              # Response combination
│   ├── inference/                          # Model inference systems
│   │   ├── vllm_integration.py             # vLLM PagedAttention inference
│   │   └── awq_quantization.py             # 4-bit AWQ quantization
│   └── core/                               # Core system components
│       ├── memory_manager.py               # Memory optimization
│       └── quantization_manager.py         # Dynamic quantization
├── templates/                              # HTML templates (Bootstrap-based)
│   ├── base.html                           # Base template with Bootstrap
│   ├── index.html                          # Main dashboard
│   ├── finetune.html                       # Fine-tuning interface
│   ├── rag.html                            # RAG interface
│   └── models.html                         # Model management
├── configs/                                # System configurations
│   ├── available_models.json               # Available model registry
│   ├── continual_tasks.yaml               # Continual learning task definitions
│   ├── default_config.yaml                # Default system configuration
│   └── deepspeed/                          # DeepSpeed configurations
│       ├── ds_config_small.json            # Small model DeepSpeed config
│       ├── ds_config_medium.json           # Medium model config
│       ├── ds_config_large.json            # Large model config
│       └── ds_config_ultra_large.json      # Ultra-large model config
├── config/                                 # Additional configuration files
│   ├── model_config.yaml                   # Model specifications
│   ├── rag_config.yaml                     # RAG system config
│   ├── continual_learning_config.yaml      # Continual learning config
│   └── ollama_config.json                  # Ollama integration config
├── scripts/                                # Automation and utility scripts
│   ├── docker_build_rag.sh                # Docker build automation
│   ├── start_web_interface.sh              # Web interface startup
│   ├── setup_environment.sh               # Environment setup
│   ├── train_large_model.py               # Large model training
│   ├── train_calm3_22b.py                 # CALM3-22B specific training
│   ├── test/                               # Testing scripts
│   │   ├── test_integration.py             # Integration testing
│   │   ├── test_docker_rag.py              # Docker RAG testing
│   │   ├── simple_lora_tutorial.py         # LoRA tutorial/demo
│   │   └── test_memory_optimization.py     # Memory optimization tests
│   ├── rag/                                # RAG-specific scripts
│   │   ├── index_documents.py              # Document indexing
│   │   └── setup_rag_env.sh                # RAG environment setup
│   ├── moe/                                # MoE-specific scripts
│   │   ├── train_moe.sh                    # MoE training automation
│   │   └── test_moe_integration.py         # MoE integration testing
│   └── continual_learning/                 # Continual learning scripts
│       ├── run_continual_learning.sh       # Continual learning execution
│       └── run_pipeline.py                 # Pipeline execution
├── docker/                                 # Docker configuration
│   ├── docker-compose.yml                 # Main Docker Compose
│   ├── docker-compose-monitoring.yml      # Monitoring stack
│   ├── docker-compose-ollama.yml          # Ollama integration
│   ├── Dockerfile                         # Main container definition
│   ├── entrypoint.sh                      # Container entrypoint
│   └── prometheus/grafana/                 # Monitoring configurations
├── data/                                   # Data storage and management
│   ├── civil_engineering/                 # Civil engineering datasets
│   │   ├── train/                          # Training data by domain
│   │   │   ├── road_design.jsonl           # Road design data
│   │   │   ├── structural_design.jsonl     # Structural engineering
│   │   │   ├── geotechnical.jsonl          # Geotechnical data
│   │   │   ├── materials.jsonl             # Materials engineering
│   │   │   ├── hydraulics.jsonl            # Hydraulic engineering
│   │   │   ├── environmental.jsonl         # Environmental engineering
│   │   │   ├── regulations.jsonl           # Regulatory compliance
│   │   │   └── construction_management.jsonl # Construction management
│   │   └── val/                            # Validation data (same structure)
│   ├── uploaded/                           # User-uploaded datasets
│   │   ├── AI_FT_data.jsonl                # Fine-tuning datasets
│   │   ├── conc_data.jsonl                 # Concrete engineering data
│   │   └── R_training_data.jsonl           # R programming data
│   ├── continual_learning/                 # Continual learning task data
│   │   ├── tasks_state.json                # Task state tracking
│   │   └── [uuid]_[dataset].jsonl          # Task-specific datasets
│   ├── rag_persistent/                     # RAG persistent storage
│   │   ├── vectors/                        # Vector database storage
│   │   ├── metadata/                       # Document metadata
│   │   └── backups/                        # System backups
│   └── qdrant/                             # Qdrant vector database files
├── docs/                                   # Project documentation
│   ├── USER_MANUAL.md                      # User manual
│   ├── API_REFERENCE.md                    # API documentation
│   ├── ROAD_DESIGN_RAG_ARCHITECTURE.md     # RAG architecture guide
│   ├── DOCKER_RAG_INTEGRATION.md           # Docker integration guide
│   ├── MULTI_GPU_OPTIMIZATION.md           # Multi-GPU optimization
│   ├── LARGE_MODEL_SETUP.md                # Large model setup guide
│   ├── PERFORMANCE_OPTIMIZATION_GUIDE.md   # Performance tuning
│   └── complete_setup/                     # Complete setup documentation
├── outputs/                                # Model outputs and checkpoints
│   ├── lora_models/                        # LoRA adapter checkpoints
│   ├── full_models/                        # Full fine-tuned models
│   ├── continual_task_*/                   # Continual learning task models
│   └── ewc_data/                           # EWC Fisher matrices and task history
├── logs/                                   # Application and training logs
│   ├── training/                           # Training logs
│   ├── rag/                                # RAG system logs
│   └── api/                                # API request logs
├── monitoring/                             # Monitoring and metrics
│   ├── prometheus/                         # Prometheus configuration
│   └── grafana/                            # Grafana dashboards
├── static/                                 # Additional static assets
│   └── logo_teikoku.png                    # Company logo
├── tests/                                  # Test suite
│   ├── unit/                               # Unit tests
│   ├── integration/                        # Integration tests
│   └── performance/                        # Performance tests
├── requirements.txt                        # Python dependencies
├── requirements-dev.txt                    # Development dependencies
├── requirements-rag.txt                    # RAG-specific dependencies
├── setup.py                                # Package setup configuration
├── pyproject.toml                          # Modern Python project config
├── .env.example                            # Environment variables template
├── .gitignore                              # Git ignore patterns
├── README.md                               # Project readme
├── CLAUDE.md                               # Claude AI assistant instructions
├── ARCHITECTURE.md                         # This file - project structure documentation
└── LICENSE                                 # Project license
```

## Module Descriptions

### Core Application (`app/`)
- **Purpose**: Main FastAPI web application providing unified interface for fine-tuning and RAG
- **Key Files**:
  - `main_unified.py`: Central FastAPI server handling all endpoints on port 8050
  - `memory_optimized_loader.py`: Dynamic model quantization and memory management
  - `model_utils.py`: Model loading utilities with GPU/CPU optimization
  - `ollama_integration.py`: Integration with Ollama for Llama 3.2 3B model
- **Dependencies**: FastAPI routers, static files, monitoring components

### Training System (`src/training/`)
- **Purpose**: Comprehensive model training infrastructure with multiple techniques
- **Key Files**:
  - `lora_finetuning.py`: Parameter-efficient LoRA fine-tuning
  - `full_finetuning.py`: Complete model fine-tuning
  - `continual_learning_pipeline.py`: EWC-based continual learning system
  - `ewc_utils.py`: Elastic Weight Consolidation implementation
  - `multi_gpu_training.py`: Distributed training across multiple GPUs
  - `dora/dora_implementation.py`: DoRA (Weight-Decomposed LoRA) implementation
- **Dependencies**: HuggingFace Transformers, DeepSpeed, PyTorch

### RAG System (`src/rag/`)
- **Purpose**: Advanced retrieval-augmented generation system for civil engineering documents
- **Key Files**:
  - `core/query_engine.py`: Main RAG processing with hybrid search
  - `indexing/vector_store.py`: Qdrant vector database integration
  - `retrieval/hybrid_search.py`: Combined vector and keyword search
  - `document_processing/pdf_processor.py`: PDF document processing with OCR
  - `specialized/numerical_processor.py`: Engineering calculation processing
- **Dependencies**: Qdrant, sentence-transformers, PyPDF2, OCR libraries

### MoE Integration (`src/moe_rag_integration/`)
- **Purpose**: Integration layer combining Mixture of Experts with RAG functionality
- **Key Files**:
  - `unified_moe_rag_system.py`: Unified MoE-RAG system implementation
  - `expert_router.py`: Expert routing logic for domain-specific queries
  - `hybrid_query_engine.py`: MoE-enhanced query processing
  - `response_fusion.py`: Multi-expert response combination
- **Dependencies**: MoE architecture, RAG core components

### Inference System (`src/inference/`)
- **Purpose**: High-performance model inference with memory optimization
- **Key Files**:
  - `vllm_integration.py`: vLLM PagedAttention for high-throughput inference
  - `awq_quantization.py`: AWQ 4-bit quantization for 75% memory reduction
- **Dependencies**: vLLM, AWQ quantization libraries

## Configuration Files

| File | Purpose |
|------|---------|
| `configs/available_models.json` | Registry of available models and their specifications |
| `config/model_config.yaml` | Model loading and quantization configurations |
| `src/rag/config/rag_config.yaml` | RAG system parameters and embedding settings |
| `configs/continual_tasks.yaml` | Continual learning task definitions |
| `configs/deepspeed/*.json` | DeepSpeed configurations for different model sizes |
| `docker/docker-compose.yml` | Main Docker container orchestration |

## Quick Navigation

### Entry Points
- **Web Interface**: `app/main_unified.py` (FastAPI server on port 8050)
- **Docker Setup**: `scripts/docker_build_rag.sh`
- **Training Scripts**: `scripts/train_large_model.py`
- **RAG Indexing**: `scripts/rag/index_documents.py`

### Key Directories for Development
- **API Development**: `app/routers/`
- **Model Training**: `src/training/`
- **RAG Implementation**: `src/rag/core/`
- **Configuration**: `configs/` and `config/`
- **Documentation**: `docs/`
- **Testing**: `scripts/test/`

### Data Management
- **Training Data**: `data/civil_engineering/`
- **Uploaded Files**: `data/uploaded/`
- **Vector Storage**: `data/rag_persistent/vectors/`
- **Continual Learning**: `data/continual_learning/`

## Special Features

### Memory Optimization
- Dynamic quantization: 32B/22B models use 4-bit, 7B/8B models use 8-bit
- CPU offloading for large models
- Model caching to reduce loading times
- AWQ quantization for 75% memory reduction

### Continual Learning
- EWC-based learning with Fisher Information Matrix
- Task history tracking in `data/continual_learning/tasks_state.json`
- Fisher matrices saved as `outputs/ewc_data/fisher_task_*.pt`
- Models saved to `outputs/continual_task_*`

### RAG Vector Store
- Qdrant with UUID-based point IDs
- Hybrid search (0.7 vector + 0.3 keyword weight)
- Multilingual-e5-large embeddings
- 512-token chunks with 128-token overlap

### Docker Integration
- Complete containerized environment
- Multi-service orchestration with monitoring
- Ollama integration for additional model support
- GPU support with NVIDIA runtime

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

## Development Workflow

### Quick Start
```bash
# 1. Build and start Docker environment
./scripts/docker_build_rag.sh --no-cache

# 2. Start web interface
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh

# 3. Access the application
# Open browser to http://localhost:8050
```

### Common Operations
```bash
# View logs
docker logs -f ai-ft-container

# Enter container
docker exec -it ai-ft-container bash

# Run tests
python scripts/test_integration.py

# Index RAG documents
python scripts/rag/index_documents.py

# Start Ollama (separate terminal)
ollama serve
```

### Development Tips
1. **Always use the Docker environment** for consistency
2. **Test changes locally** before deploying using the test scripts
3. **Monitor logs** during development: `docker logs -f ai-ft-container`
4. **Use the unified API** at port 8050 rather than separate services
5. **Check GPU memory** before training large models: `nvidia-smi`
6. **Backup trained models** from the `outputs/` directory regularly
7. **Review task history** in `data/continual_learning/tasks_state.json` for continual learning
8. **Keep Ollama running** in a separate terminal for Ollama model integration

## Project Highlights

This comprehensive structure supports both research and production deployments, with clear separation of concerns and modular architecture enabling independent development of fine-tuning, RAG, and MoE components. The project is designed for:

- **Scalability**: Multi-GPU support, distributed training, and efficient inference
- **Flexibility**: Multiple training techniques (LoRA, DoRA, EWC, full fine-tuning)
- **Domain Specialization**: Civil engineering and road design focus with specialized processing
- **Production Ready**: Docker containerization, monitoring, and robust error handling
- **Memory Efficiency**: Dynamic quantization and AWQ for reduced memory footprint
- **Continuous Learning**: EWC-based continual learning for evolving knowledge bases