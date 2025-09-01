# Critical Files and Their Purposes

## Core Application Files

### `app/main_unified.py`
- **Purpose**: Main FastAPI application server
- **Key Classes**: RAGApplication, ModelInfo, TrainingRequest
- **Responsibilities**:
  - Route handling for all APIs
  - Background task management
  - Model loading coordination
  - WebSocket connections

### `app/memory_optimized_loader.py`
- **Purpose**: Dynamic model loading with memory optimization
- **Functions**: Load models with appropriate quantization
- **Features**: Auto-detection of model size, CPU offloading

## Training System

### `src/training/lora_finetuning.py`
- **Purpose**: LoRA parameter-efficient fine-tuning
- **Key Functions**: setup_lora_model, train_lora

### `src/training/continual_learning_pipeline.py`
- **Purpose**: EWC-based continual learning
- **Features**: Fisher matrix calculation, task management

### `src/training/ewc_utils.py`
- **Purpose**: Elastic Weight Consolidation utilities
- **Functions**: compute_fisher_matrix, ewc_loss

## RAG System

### `src/rag/core/query_engine.py`
- **Purpose**: Main RAG query processing
- **Classes**: QueryEngine, QueryProcessor
- **Pipeline**: Query → Search → Re-rank → Generate

### `src/rag/indexing/vector_store.py`
- **Purpose**: Qdrant vector database interface
- **Functions**: add_documents, search, delete

### `src/rag/document_processing/pdf_processor.py`
- **Purpose**: PDF document processing
- **Features**: OCR, table extraction, text chunking

## Configuration Files

### `docker/docker-compose.yml`
- **Purpose**: Multi-container orchestration
- **Services**: ai-ft, qdrant, tensorboard, jupyter

### `docker/Dockerfile`
- **Purpose**: Container image definition
- **Base**: pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

### `src/rag/config/rag_config.yaml`
- **Purpose**: RAG system configuration
- **Settings**: Embedding model, chunk size, search weights

### `config/model_config.yaml`
- **Purpose**: Model specifications
- **Content**: Model names, paths, quantization settings

## Utility Scripts

### `scripts/docker_build_rag.sh`
- **Purpose**: Build Docker environment with RAG
- **Features**: Dependency check, image build, test run

### `scripts/start_web_interface.sh`
- **Purpose**: Start unified web server
- **Actions**: Permission setup, Ollama init, server launch

### `scripts/rag/index_documents.py`
- **Purpose**: Index documents into vector database
- **Process**: Load PDFs → Process → Embed → Store

## Templates

### `templates/base.html`
- **Purpose**: Base template with navigation
- **Features**: Bootstrap 5, common JS/CSS

### `templates/rag.html`
- **Purpose**: RAG interface
- **Features**: Document upload, query interface, results display

### `templates/finetune.html`
- **Purpose**: Fine-tuning interface
- **Features**: Model selection, training config, progress tracking

### `templates/continual.html`
- **Purpose**: Continual learning management
- **Features**: Task creation, history view, model selection

## Data Files

### `data/continual_learning/tasks_state.json`
- **Purpose**: Continual learning task tracking
- **Content**: Task status, timestamps, configurations

### `outputs/ewc_data/task_history.json`
- **Purpose**: EWC task history
- **Content**: Completed tasks, Fisher matrices references

## Environment Files

### `.env.example`
- **Purpose**: Environment variable template
- **Variables**: HF_TOKEN, WANDB_API_KEY, CUDA_VISIBLE_DEVICES

### `requirements.txt`
- **Purpose**: Python dependencies
- **Categories**: ML frameworks, web, utilities

### `requirements_rag.txt`
- **Purpose**: RAG-specific dependencies
- **Packages**: langchain, llama-index, qdrant-client