# Ollama Models Directory

This directory contains Modelfile configurations for Ollama custom models.

## Available Models

### road_engineering_expert
- **Base Model**: llama3.2:3b
- **Purpose**: 道路工学専門の質問応答
- **Language**: Japanese
- **Parameters**:
  - temperature: 0.7
  - top_p: 0.9
  - top_k: 40
  - repeat_penalty: 1.1

## Usage

To create the model in Ollama:
```bash
ollama create road_engineering_expert -f road_engineering_expert.Modelfile
```

To use the model:
```bash
ollama run road_engineering_expert
```

## Notes
- Models require Ollama to be installed and running (port 11434)
- Base model (llama3.2:3b) must be downloaded first: `ollama pull llama3.2:3b`