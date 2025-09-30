# MoE-RAG Documentation Index

Welcome to the MoE-RAG documentation. This guide will help you navigate the documentation based on your needs.

## 🚀 Getting Started

New to MoE-RAG? Start here:

1. [**README.md**](../README.md) - Quick start guide and project overview
2. [**Quick Start**](../README.md#quick-start) - Get the system running in 3 steps
3. [**System Overview**](../README.md#system-overview) - Understand the three core systems

## 📚 Core Documentation

### System Architecture
- [**ARCHITECTURE.md**](ARCHITECTURE.md) - Complete system architecture
  - Layered architecture design
  - Data flow patterns
  - Service integration
  - Deployment strategies

### API Reference
- [**API_REFERENCE.md**](API_REFERENCE.md) - Complete REST API documentation
  - Fine-tuning API endpoints
  - RAG API endpoints
  - Continual Learning API endpoints
  - Model Management API
  - WebSocket APIs

### User Guides
- [**USER_MANUAL.md**](USER_MANUAL.md) - End-user operations manual
  - Web interface usage
  - Common workflows
  - Troubleshooting

## 🔧 Feature-Specific Guides

### Fine-tuning
- [**TRAINED_MODEL_USAGE.md**](TRAINED_MODEL_USAGE.md) - Working with trained models
- [**GPT_NEOX_DYNAMIC_LORA_USAGE.md**](GPT_NEOX_DYNAMIC_LORA_USAGE.md) - Dynamic LoRA implementation
- [**LARGE_MODEL_SETUP.md**](LARGE_MODEL_SETUP.md) - Training large models (32B+)

### RAG System
- [**RAG_HYBRID_SEARCH_GUIDE.md**](RAG_HYBRID_SEARCH_GUIDE.md) - Vector + keyword search
- [**ROAD_DESIGN_RAG_ARCHITECTURE.md**](ROAD_DESIGN_RAG_ARCHITECTURE.md) - Road design specialization
- [**gguf_rag_integration.md**](gguf_rag_integration.md) - GGUF model integration

### Continual Learning
- [**continual_learning_guide.md**](continual_learning_guide.md) - EWC-based continual learning
  - Task management
  - Fisher Information Matrix
  - Catastrophic forgetting prevention

## ⚙️ System Configuration

### Docker & Deployment
- [**DOCKER_RAG_INTEGRATION.md**](DOCKER_RAG_INTEGRATION.md) - Docker setup and integration
- [**container_setup/**](container_setup/) - Container configuration
- [**complete_setup/**](complete_setup/) - Complete deployment scripts

### Performance & Optimization
- [**PERFORMANCE_OPTIMIZATION_GUIDE.md**](PERFORMANCE_OPTIMIZATION_GUIDE.md) - Comprehensive optimization
- [**MULTI_GPU_OPTIMIZATION.md**](MULTI_GPU_OPTIMIZATION.md) - Multi-GPU training
- [**model_quantization_for_ollama.md**](model_quantization_for_ollama.md) - GGUF quantization

### Integration
- [**WEB_INTERFACE_INTEGRATION.md**](WEB_INTERFACE_INTEGRATION.md) - Web UI integration
- [**GPT_NEOX_LORA_INTEGRATION.md**](GPT_NEOX_LORA_INTEGRATION.md) - GPT-NeoX LoRA
- [**OLLAMA_MODEL_SYNC_SOLUTION.md**](OLLAMA_MODEL_SYNC_SOLUTION.md) - Ollama synchronization

## 📊 Monitoring & Operations

- [**MONITORING_GRAFANA_GUIDE.md**](MONITORING_GRAFANA_GUIDE.md) - Grafana monitoring setup
- [**DEPENDENCY_MANAGEMENT.md**](DEPENDENCY_MANAGEMENT.md) - Dependency tracking

## 🔍 Technical Analysis

Internal technical documentation for advanced users and developers:

- [**rag_system_structure_analysis.md**](rag_system_structure_analysis.md) - RAG system internals
- [**model_loading_flow_analysis.md**](model_loading_flow_analysis.md) - Model loading pipeline
- [**system_flow_verification.md**](system_flow_verification.md) - System workflow validation
- [**deepseek_32b_quantization_verification.md**](deepseek_32b_quantization_verification.md) - Quantization analysis
- [**finetuned_model_quantization_analysis.md**](finetuned_model_quantization_analysis.md) - Fine-tuned model quantization
- [**qlora_rag_workflow.md**](qlora_rag_workflow.md) - QLoRA workflow analysis

## 🚦 Production Deployment

- [**NEXT_STEPS_PRODUCTION.md**](NEXT_STEPS_PRODUCTION.md) - Production deployment checklist
  - Scaling strategies
  - Security hardening
  - Backup and recovery
  - Monitoring setup

## 📖 By Use Case

### I want to train a model
1. Read [README.md Quick Start](../README.md#quick-start)
2. Check [ARCHITECTURE.md - Fine-tuning Services](ARCHITECTURE.md#fine-tuning-services)
3. Use [API_REFERENCE.md - Fine-tuning API](API_REFERENCE.md#fine-tuning-api)
4. For large models, see [LARGE_MODEL_SETUP.md](LARGE_MODEL_SETUP.md)

### I want to use RAG for document search
1. Read [README.md RAG System](../README.md#rag-system)
2. Check [RAG_HYBRID_SEARCH_GUIDE.md](RAG_HYBRID_SEARCH_GUIDE.md)
3. Use [API_REFERENCE.md - RAG API](API_REFERENCE.md#rag-api)
4. For road design, see [ROAD_DESIGN_RAG_ARCHITECTURE.md](ROAD_DESIGN_RAG_ARCHITECTURE.md)

### I want to set up continual learning
1. Read [continual_learning_guide.md](continual_learning_guide.md)
2. Check [ARCHITECTURE.md - Continual Learning Flow](ARCHITECTURE.md#continual-learning-flow)
3. Use [API_REFERENCE.md - Continual Learning API](API_REFERENCE.md#continual-learning-api)

### I want to deploy to production
1. Read [DOCKER_RAG_INTEGRATION.md](DOCKER_RAG_INTEGRATION.md)
2. Check [NEXT_STEPS_PRODUCTION.md](NEXT_STEPS_PRODUCTION.md)
3. Set up [MONITORING_GRAFANA_GUIDE.md](MONITORING_GRAFANA_GUIDE.md)
4. Review [PERFORMANCE_OPTIMIZATION_GUIDE.md](PERFORMANCE_OPTIMIZATION_GUIDE.md)

### I want to optimize performance
1. Start with [PERFORMANCE_OPTIMIZATION_GUIDE.md](PERFORMANCE_OPTIMIZATION_GUIDE.md)
2. For multi-GPU, see [MULTI_GPU_OPTIMIZATION.md](MULTI_GPU_OPTIMIZATION.md)
3. For quantization, see [model_quantization_for_ollama.md](model_quantization_for_ollama.md)

## 🆘 Troubleshooting

Having issues? Check these resources:

1. [README.md Troubleshooting](../README.md#troubleshooting) - Common issues
2. [USER_MANUAL.md](USER_MANUAL.md) - User guide
3. [API_REFERENCE.md Error Codes](API_REFERENCE.md#common-response-codes) - API errors
4. [GitHub Issues](https://github.com/kji-furuta/MoE_RAG/issues) - Report bugs

## 🔄 Version History

- **v4.0.0 (2025-09-30)** - Current version
  - ✅ All three core systems operational
  - ✅ Unified API server (port 8050)
  - ✅ Ollama integration with automatic GGUF detection
  - ✅ EWC-based continual learning
  - ✅ Hybrid RAG search

## 📞 Support & Resources

- **GitHub Repository**: https://github.com/kji-furuta/MoE_RAG
- **Issues**: https://github.com/kji-furuta/MoE_RAG/issues
- **Documentation**: https://github.com/kji-furuta/MoE_RAG/tree/main/docs

---

**Need help?** Start with the [README.md](../README.md) for quick guidance, or browse the [API Reference](API_REFERENCE.md) for detailed endpoint documentation.