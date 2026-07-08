# MoE-RAG Project Overview

## Project Purpose
AI Fine-tuning Toolkit (MoE-RAG) is a comprehensive Japanese LLM fine-tuning platform with integrated RAG (Retrieval-Augmented Generation) system specialized for civil engineering and road design. The project provides a unified web interface for both fine-tuning and RAG functionality through a single port (8050).

## Key Features
1. **Unified Web Interface**: Single port access (8050) for all features
2. **Fine-tuning System**: LoRA, DoRA, and full fine-tuning support
3. **RAG System**: Vector search with Qdrant, specialized for civil engineering documents
4. **Continual Learning**: EWC-based system to prevent catastrophic forgetting
5. **MoE (Mixture of Experts)**: Expert models for specialized domains
6. **Ollama Integration**: Support for Llama 3.2 3B model

## Project Structure
```
MoE_RAG/
├── app/                    # Web application (FastAPI)
│   ├── main_unified.py    # Main server file
│   └── memory_optimized_loader.py  # Model loading
├── src/                   # Core source code
│   ├── training/         # Fine-tuning modules
│   ├── rag/             # RAG system
│   └── inference/       # Inference engines
├── docker/              # Docker configuration
├── scripts/            # Utility scripts
├── templates/          # HTML templates
├── data/              # Training and RAG data
├── outputs/           # Model outputs
└── config/           # Configuration files
```

## GitHub Repository
- **Main Repository**: https://github.com/kji-furuta/MoE_RAG.git
- **Remote name**: moe_rag
- **Main branch**: main
- **Push command**: `git push moe_rag main`