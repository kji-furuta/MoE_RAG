# Code Style and Conventions

## Python Code Style
- **Formatter**: Black with line-length=88
- **Import Sorting**: isort with profile="black"
- **Target Python Version**: 3.8+
- **Docstring Style**: Google-style docstrings

## Naming Conventions
- **Classes**: PascalCase (e.g., `RAGApplication`, `ModelInfo`)
- **Functions**: snake_case (e.g., `run_training_task`, `get_system_info`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `RAG_AVAILABLE`, `OLLAMA_AVAILABLE`)
- **Private methods**: Leading underscore (e.g., `_initialize_model`)
- **File names**: snake_case (e.g., `main_unified.py`, `memory_optimized_loader.py`)

## Type Hints
- Use type hints for function parameters and return values
- Use Pydantic models for API request/response schemas
- Example:
```python
from typing import Optional, List, Dict
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    model_name: Optional[str] = None
```

## Error Handling
- Use try-except blocks with specific exception types
- Log errors with appropriate levels (DEBUG, INFO, WARNING, ERROR)
- Return meaningful error messages in API responses
- Example:
```python
try:
    result = process_query(query)
except ValueError as e:
    logger.error(f"Invalid query: {e}")
    raise HTTPException(status_code=400, detail=str(e))
```

## Async/Await Patterns
- Use async/await for I/O operations
- Background tasks for long-running operations
- AsyncIO for concurrent processing

## File Organization
- Keep related functionality in dedicated modules
- Separate concerns (models, views, controllers)
- Use `__init__.py` to expose public APIs
- Configuration in YAML files

## Comments and Documentation
- Japanese comments for business logic (日本語コメント使用)
- English comments for technical implementation
- Comprehensive README files
- API documentation in code