# MoE-RAG API Reference

## Base URL

```
http://localhost:8050
```

All API endpoints are served from a single unified FastAPI server.

---

## Table of Contents

1. [Fine-tuning API](#fine-tuning-api)
2. [RAG API](#rag-api)
3. [Continual Learning API](#continual-learning-api)
4. [Model Management API](#model-management-api)
5. [System API](#system-api)
6. [Common Response Codes](#common-response-codes)

---

## Fine-tuning API

### Start Training

**POST** `/api/train`

Start a new fine-tuning job.

<details>
<summary><b>Request Body</b></summary>

```json
{
  "model_name": "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
  "training_file": "data/training_data.jsonl",
  "validation_file": "data/validation_data.jsonl",
  "output_dir": "outputs/my_model",
  "learning_rate": 2e-5,
  "num_train_epochs": 3,
  "batch_size": 1,
  "gradient_accumulation_steps": 16,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
  "use_dora": false,
  "fp16": false,
  "mixed_precision": "bf16"
}
```

**Parameters**:
- `model_name` (string, required): Base model identifier
- `training_file` (string, required): Path to training data (JSONL format)
- `validation_file` (string, optional): Path to validation data
- `output_dir` (string, required): Output directory for checkpoints
- `learning_rate` (float): Learning rate (default: 2e-5)
- `num_train_epochs` (int): Number of training epochs (default: 3)
- `batch_size` (int): Per-device batch size (default: 1)
- `gradient_accumulation_steps` (int): Gradient accumulation steps (default: 16)
- `lora_r` (int): LoRA rank (default: 16)
- `lora_alpha` (int): LoRA alpha (default: 32)
- `lora_dropout` (float): LoRA dropout (default: 0.05)
- `target_modules` (array[string]): Target modules for LoRA
- `use_dora` (bool): Use DoRA instead of LoRA (default: false)
- `fp16` (bool): Use FP16 precision (default: false)
- `mixed_precision` (string): Mixed precision mode ("bf16", "fp16", "no")

</details>

<details>
<summary><b>Response</b></summary>

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "started"
}
```

**Status Codes**:
- `200`: Training started successfully
- `400`: Invalid request parameters
- `500`: Internal server error

</details>

---

### Get Training Status

**GET** `/api/training-status/{task_id}`

Get the status and progress of a training job.

<details>
<summary><b>Path Parameters</b></summary>

- `task_id` (string, required): Training task UUID

</details>

<details>
<summary><b>Response</b></summary>

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "running",
  "progress": 45,
  "current_step": 450,
  "total_steps": 1000,
  "loss": 0.234,
  "learning_rate": 1.5e-5,
  "output_dir": "outputs/my_model",
  "started_at": "2025-09-30T10:30:00+09:00",
  "estimated_completion": "2025-09-30T12:00:00+09:00"
}
```

**Status Values**:
- `pending`: Waiting to start
- `running`: Currently training
- `completed`: Successfully finished
- `failed`: Training failed
- `cancelled`: User cancelled

</details>

---

### Generate Text

**POST** `/api/generate`

Generate text using a trained model.

<details>
<summary><b>Request Body</b></summary>

```json
{
  "model_path": "outputs/my_model/checkpoint-final",
  "prompt": "道路の設計速度が80km/hの場合、",
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "do_sample": true
}
```

**Parameters**:
- `model_path` (string, required): Path to model checkpoint
- `prompt` (string, required): Input text prompt
- `max_new_tokens` (int): Maximum tokens to generate (default: 512)
- `temperature` (float): Sampling temperature (default: 0.7)
- `top_p` (float): Nucleus sampling threshold (default: 0.9)
- `do_sample` (bool): Enable sampling (default: true)

</details>

<details>
<summary><b>Response</b></summary>

```json
{
  "generated_text": "道路の設計速度が80km/hの場合、最小曲線半径は280mとなります。",
  "model_path": "outputs/my_model/checkpoint-final",
  "generation_time_seconds": 1.234
}
```

</details>

---

## RAG API

### Query Documents

**POST** `/rag/query`

Search and answer questions using RAG.

<details>
<summary><b>Request Body</b></summary>

```json
{
  "query": "設計速度80km/hの道路の最小曲線半径は？",
  "top_k": 5,
  "search_type": "hybrid",
  "model": "ollama:deepseek-32b-finetuned:latest",
  "filters": {
    "document_type": "道路構造令",
    "chapter": "第3章"
  },
  "document_ids": ["uuid-1", "uuid-2"],
  "include_sources": true
}
```

**Parameters**:
- `query` (string, required): Search query or question
- `top_k` (int): Number of chunks to retrieve (default: 5)
- `search_type` (string): Search algorithm
  - `hybrid`: Vector + keyword (default)
  - `vector`: Vector search only
  - `keyword`: BM25 keyword only
- `model` (string, optional): Override default LLM
  - Format: `ollama:<model_name>`
  - Format: `finetuned:<model_path>`
- `filters` (object, optional): Metadata filters
- `document_ids` (array[string], optional): Restrict to specific documents
- `include_sources` (bool): Include source citations (default: true)

</details>

<details>
<summary><b>Response</b></summary>

```json
{
  "answer": "設計速度80km/hの道路における最小曲線半径は280mです。",
  "sources": [
    {
      "document_id": "uuid-1",
      "document_name": "道路構造令.pdf",
      "chunk_id": "chunk-123",
      "content": "第三章 第三条 設計速度80km/hにおける曲線半径は280m以上とする。",
      "page_number": 15,
      "relevance_score": 0.92,
      "metadata": {
        "chapter": "第3章",
        "section": "第3条"
      }
    }
  ],
  "query_time_seconds": 0.876,
  "model_used": "ollama:deepseek-32b-finetuned:latest"
}
```

</details>

---

### Stream Query

**POST** `/rag/stream-query`

Stream RAG responses using Server-Sent Events (SSE).

<details>
<summary><b>Request Body</b></summary>

Same as `/rag/query` but returns streaming response.

</details>

<details>
<summary><b>Response</b></summary>

**Content-Type**: `text/event-stream`

```
data: {"type": "search", "status": "retrieving", "top_k": 5}

data: {"type": "search", "status": "completed", "num_results": 5}

data: {"type": "generation", "status": "started"}

data: {"type": "token", "content": "設計速度"}

data: {"type": "token", "content": "80km/h"}

data: {"type": "generation", "status": "completed"}

data: {"type": "sources", "sources": [...]}
```

**Event Types**:
- `search`: Document retrieval status
- `generation`: LLM generation status
- `token`: Generated token (streaming)
- `sources`: Source citations
- `error`: Error message

</details>

---

### Upload Document

**POST** `/rag/upload-document`

Upload and index a new document.

<details>
<summary><b>Request</b></summary>

**Content-Type**: `multipart/form-data`

**Form Fields**:
- `file` (file, required): PDF/TXT document
- `document_name` (string, optional): Custom document name
- `metadata` (string, optional): JSON metadata object
- `chunk_size` (int, optional): Override chunk size
- `chunk_overlap` (int, optional): Override chunk overlap

**Example** (curl):
```bash
curl -X POST http://localhost:8050/rag/upload-document \
  -F "file=@/path/to/document.pdf" \
  -F "document_name=道路構造令" \
  -F 'metadata={"type":"法令","version":"令和3年"}'
```

</details>

<details>
<summary><b>Response</b></summary>

```json
{
  "document_id": "uuid-1234",
  "document_name": "道路構造令.pdf",
  "status": "processing",
  "chunks_created": 0,
  "estimated_time_seconds": 120,
  "upload_id": "upload-5678"
}
```

</details>

---

### List Documents

**GET** `/rag/documents`

List all indexed documents.

<details>
<summary><b>Query Parameters</b></summary>

- `skip` (int): Pagination offset (default: 0)
- `limit` (int): Max results (default: 100)
- `document_type` (string, optional): Filter by type

</details>

<details>
<summary><b>Response</b></summary>

```json
{
  "documents": [
    {
      "document_id": "uuid-1",
      "document_name": "道路構造令.pdf",
      "chunk_count": 156,
      "upload_date": "2025-09-30T10:00:00+09:00",
      "file_size_bytes": 2048576,
      "metadata": {
        "type": "法令",
        "version": "令和3年"
      }
    }
  ],
  "total": 1,
  "skip": 0,
  "limit": 100
}
```

</details>

---

### Delete Document

**DELETE** `/rag/documents/{document_id}`

Delete a document and all its chunks.

<details>
<summary><b>Path Parameters</b></summary>

- `document_id` (string, required): Document UUID

</details>

<details>
<summary><b>Response</b></summary>

```json
{
  "status": "deleted",
  "document_id": "uuid-1",
  "chunks_deleted": 156
}
```

</details>

---

### Health Check

**GET** `/rag/health`

Check RAG system health.

<details>
<summary><b>Response</b></summary>

```json
{
  "status": "healthy",
  "components": {
    "vector_store": {
      "status": "connected",
      "collection": "road_design_docs",
      "point_count": 15678
    },
    "embedding_model": {
      "status": "loaded",
      "model": "intfloat/multilingual-e5-large",
      "dimension": 1024
    },
    "llm": {
      "status": "available",
      "provider": "ollama",
      "model": "deepseek-32b-finetuned:latest"
    }
  },
  "uptime_seconds": 86400
}
```

</details>

---

## Continual Learning API

### Start Continual Learning Task

**POST** `/api/continual/train`

Start a new continual learning task with EWC.

<details>
<summary><b>Request Body</b></summary>

```json
{
  "task_name": "task_11",
  "base_model": "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
  "previous_model_path": "outputs/continual_task_10_20250930_031616/checkpoint-final",
  "training_file": "data/continual/task_11_data.jsonl",
  "output_dir": "outputs/continual_task_11",
  "ewc_lambda": 5000,
  "learning_rate": 1e-5,
  "num_train_epochs": 3,
  "lora_r": 16,
  "lora_alpha": 32
}
```

**Parameters**:
- `task_name` (string, required): Task identifier
- `base_model` (string, required): Base model or previous task model
- `previous_model_path` (string, optional): Path to previous task checkpoint
- `training_file` (string, required): New task training data
- `output_dir` (string, required): Output directory
- `ewc_lambda` (float): EWC regularization strength (default: 5000)
- `learning_rate` (float): Learning rate (default: 1e-5)
- `num_train_epochs` (int): Training epochs (default: 3)
- `lora_r` (int): LoRA rank (default: 16)
- `lora_alpha` (int): LoRA alpha (default: 32)

</details>

<details>
<summary><b>Response</b></summary>

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "task_name": "task_11",
  "status": "started",
  "ewc_enabled": true,
  "num_previous_tasks": 10
}
```

</details>

---

### List Tasks

**GET** `/api/continual/tasks`

List all continual learning tasks.

<details>
<summary><b>Response</b></summary>

```json
{
  "tasks": [
    {
      "task_id": "uuid-1",
      "task_name": "task_10",
      "type": "continual_learning",
      "status": "completed",
      "progress": 100,
      "output_path": "outputs/continual_task_10_20250930_031616",
      "model_info": {
        "path": "outputs/continual_task_10_20250930_031616/checkpoint-final",
        "base_model": "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
        "training_params": {
          "lora_rank": 16,
          "lora_alpha": 32,
          "format": "safetensors"
        }
      },
      "created_at": "2025-09-30T03:16:16+09:00",
      "completed_at": "2025-09-30T05:45:30+09:00"
    }
  ],
  "total": 19
}
```

</details>

---

### Get Task Status

**GET** `/api/continual/task/{task_id}`

Get detailed status of a specific task.

<details>
<summary><b>Path Parameters</b></summary>

- `task_id` (string, required): Task UUID

</details>

<details>
<summary><b>Response</b></summary>

```json
{
  "task_id": "uuid-1",
  "task_name": "task_10",
  "status": "completed",
  "progress": 100,
  "current_step": 1000,
  "total_steps": 1000,
  "ewc_info": {
    "enabled": true,
    "lambda": 5000,
    "num_previous_tasks": 9,
    "fisher_matrix_path": "outputs/ewc_data/fisher_task_10.pt"
  },
  "model_info": {
    "path": "outputs/continual_task_10_20250930_031616/checkpoint-final",
    "format": "safetensors",
    "size_mb": 1024
  },
  "training_metrics": {
    "final_loss": 0.456,
    "best_loss": 0.423,
    "training_time_seconds": 8934
  }
}
```

</details>

---

### Update Models List

**POST** `/api/continual/update-models`

Refresh the list of available models from Ollama and filesystem.

<details>
<summary><b>Response</b></summary>

```json
{
  "status": "updated",
  "ollama_models": [
    "deepseek-32b-finetuned:latest",
    "gpt-neox-20b-finetuned:latest"
  ],
  "local_models": [
    "outputs/continual_task_10_20250930_031616/checkpoint-final"
  ],
  "total_models": 15
}
```

</details>

---

## Model Management API

### List Models

**GET** `/api/models`

List all available models (fine-tuned, Ollama, base).

<details>
<summary><b>Query Parameters</b></summary>

- `type` (string, optional): Filter by type
  - `finetuned`: Local fine-tuned models
  - `ollama`: Ollama models
  - `base`: Base HuggingFace models

</details>

<details>
<summary><b>Response</b></summary>

```json
{
  "models": [
    {
      "name": "deepseek-32b-finetuned",
      "type": "ollama",
      "tag": "latest",
      "size_gb": 18.4,
      "modified_at": "2025-09-30T10:00:00+09:00"
    },
    {
      "name": "task_10_model",
      "type": "finetuned",
      "path": "outputs/continual_task_10_20250930_031616/checkpoint-final",
      "format": "safetensors",
      "size_mb": 1024
    }
  ],
  "total": 15
}
```

</details>

---

### Convert to Ollama

**POST** `/api/convert-to-ollama`

Convert a fine-tuned model to Ollama GGUF format.

<details>
<summary><b>Request Body</b></summary>

```json
{
  "model_path": "outputs/my_model/checkpoint-final",
  "model_name": "my-custom-model",
  "quantization": "q4_k_m"
}
```

**Parameters**:
- `model_path` (string, required): Path to fine-tuned model
- `model_name` (string, required): Name for Ollama model
- `quantization` (string): GGUF quantization level
  - `q4_k_m`: 4-bit (default, balanced)
  - `q5_k_m`: 5-bit (better quality)
  - `q8_0`: 8-bit (highest quality)

</details>

<details>
<summary><b>Response</b></summary>

```json
{
  "status": "converting",
  "conversion_id": "conv-1234",
  "estimated_time_minutes": 15,
  "output_model": "my-custom-model:latest"
}
```

</details>

---

### Delete Model

**DELETE** `/api/models/{model_name}`

Delete a local fine-tuned model.

<details>
<summary><b>Path Parameters</b></summary>

- `model_name` (string, required): Model directory name

</details>

<details>
<summary><b>Response</b></summary>

```json
{
  "status": "deleted",
  "model_name": "my_model",
  "path": "outputs/my_model",
  "freed_space_mb": 1024
}
```

</details>

---

### Delete Ollama Model

**DELETE** `/api/ollama-models/{model_name}`

Delete an Ollama model.

<details>
<summary><b>Path Parameters</b></summary>

- `model_name` (string, required): Ollama model name with tag

**Example**: `deepseek-32b-finetuned:latest`

</details>

<details>
<summary><b>Response</b></summary>

```json
{
  "status": "deleted",
  "model_name": "deepseek-32b-finetuned:latest",
  "freed_space_gb": 18.4
}
```

</details>

---

## System API

### System Info

**GET** `/api/system-info`

Get system resource information.

<details>
<summary><b>Response</b></summary>

```json
{
  "system": {
    "os": "Linux",
    "python_version": "3.10.12",
    "cuda_available": true,
    "cuda_version": "12.1",
    "pytorch_version": "2.1.0"
  },
  "gpu": [
    {
      "id": 0,
      "name": "NVIDIA A100 80GB PCIe",
      "memory_total_gb": 81.92,
      "memory_used_gb": 45.67,
      "memory_free_gb": 36.25,
      "utilization_percent": 65
    }
  ],
  "cpu": {
    "cores": 32,
    "usage_percent": 23
  },
  "memory": {
    "total_gb": 128,
    "used_gb": 87,
    "free_gb": 41
  },
  "disk": {
    "total_gb": 2048,
    "used_gb": 1234,
    "free_gb": 814
  }
}
```

</details>

---

### Metrics

**GET** `/api/metrics`

Get system metrics (Prometheus-compatible format planned).

<details>
<summary><b>Response</b></summary>

```json
{
  "training": {
    "total_jobs": 156,
    "active_jobs": 2,
    "completed_jobs": 149,
    "failed_jobs": 5
  },
  "rag": {
    "total_queries": 12345,
    "avg_query_time_ms": 876,
    "documents_indexed": 234,
    "total_chunks": 156789
  },
  "continual_learning": {
    "total_tasks": 19,
    "completed_tasks": 18,
    "failed_tasks": 1
  }
}
```

</details>

---

### Health Check

**GET** `/health`

Overall system health check.

<details>
<summary><b>Response</b></summary>

```json
{
  "status": "healthy",
  "version": "4.0.0",
  "services": {
    "api": "healthy",
    "rag": "healthy",
    "ollama": "healthy",
    "qdrant": "healthy"
  },
  "uptime_seconds": 86400,
  "timestamp": "2025-09-30T12:00:00+09:00"
}
```

</details>

---

## Common Response Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Access denied |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource conflict (e.g., duplicate name) |
| 422 | Unprocessable Entity | Validation error |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

## Error Response Format

All error responses follow this format:

```json
{
  "detail": "Error message describing what went wrong",
  "error_code": "VALIDATION_ERROR",
  "timestamp": "2025-09-30T12:00:00+09:00"
}
```

## Rate Limiting

Currently not implemented. Planned for future releases:
- 100 requests/minute per IP for query endpoints
- 10 requests/minute per IP for training endpoints
- 1000 requests/minute per IP for health/info endpoints

## Authentication

Currently not implemented. Planned JWT-based authentication:

```bash
# Future authentication flow
curl -X POST http://localhost:8050/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# Response
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 3600
}

# Use token
curl -X POST http://localhost:8050/api/train \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc..." \
  -d '{"model_name": "...", ...}'
```

## WebSocket API

### Training Progress

**WebSocket** `/ws/training/{task_id}`

Real-time training progress updates.

**Example** (JavaScript):
```javascript
const ws = new WebSocket('ws://localhost:8050/ws/training/' + taskId);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Progress:', data.progress, '%');
  console.log('Loss:', data.loss);
  console.log('Step:', data.step, '/', data.total_steps);
};
```

### Continual Learning Progress

**WebSocket** `/api/continual/ws/{task_id}`

Real-time continual learning updates.

## SDK Examples

### Python

```python
import requests

# Start training
response = requests.post('http://localhost:8050/api/train', json={
    'model_name': 'cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese',
    'training_file': 'data/training.jsonl',
    'output_dir': 'outputs/my_model',
    'num_train_epochs': 3
})
task_id = response.json()['task_id']

# RAG query
response = requests.post('http://localhost:8050/rag/query', json={
    'query': '設計速度80km/hの最小曲線半径は？',
    'top_k': 5
})
answer = response.json()['answer']
```

### cURL

```bash
# Start training
curl -X POST http://localhost:8050/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
    "training_file": "data/training.jsonl",
    "output_dir": "outputs/my_model"
  }'

# RAG query
curl -X POST http://localhost:8050/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "設計速度80km/hの最小曲線半径は？",
    "top_k": 5
  }'

# Upload document
curl -X POST http://localhost:8050/rag/upload-document \
  -F "file=@/path/to/document.pdf" \
  -F "document_name=道路構造令"
```

## Changelog

### v4.0.0 (2025-09-30)
- ✅ All three core systems operational (Fine-tuning, RAG, Continual Learning)
- ✅ Unified API server on port 8050
- ✅ Ollama integration with automatic GGUF detection
- ✅ EWC-based continual learning with task state persistence
- ✅ Hybrid RAG search (vector + keyword)
- ✅ WebSocket support for real-time progress

---

For more information, see:
- [README.md](../README.md) - Quick start and overview
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [GitHub Repository](https://github.com/kji-furuta/MoE_RAG)