# Task Completion Commands

When you complete a coding task, execute these commands in order:

## 1. Code Quality Checks

### Format Code (if modified)
```bash
# Format Python files with Black
black <modified_files> --line-length 88

# Sort imports
isort <modified_files> --profile black
```

### Lint Check
```bash
# Check for code style issues
flake8 <modified_files>
```

## 2. Testing

### Run Unit Tests (if test files exist)
```bash
# Run specific test file
pytest tests/test_<feature>.py -v

# Run all tests
pytest tests/
```

### Integration Tests (for major changes)
```bash
python scripts/test_integration.py
```

## 3. Verify Changes in Docker

### If modified app/ or src/ files:
```bash
# Restart the web interface
docker exec ai-ft-container pkill -f uvicorn
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh
```

### Check Application Health
```bash
curl http://localhost:8050/rag/health
curl http://localhost:8050/api/system-info
```

## 4. Documentation Updates

### Update CLAUDE.md if:
- Added new API endpoints
- Changed major functionality
- Modified development workflow

### Update README.md if:
- Added new features
- Changed installation/setup process
- Modified user-facing functionality

## 5. Git Commit

### Stage and Commit
```bash
git add <modified_files>
git status  # Verify changes
git commit -m "feat/fix/chore: Descriptive message"
```

## 6. Verification Checklist

Before marking task complete, verify:
- [ ] Code is formatted (Black + isort)
- [ ] No linting errors
- [ ] Tests pass (if applicable)
- [ ] Docker environment still works
- [ ] API endpoints respond correctly
- [ ] No import errors in logs
- [ ] GPU memory usage is reasonable (if training)

## Common Issues to Check

### Module Import Errors
```bash
docker exec ai-ft-container python -c "import <module_name>"
```

### File Permissions
```bash
docker exec ai-ft-container ls -la /workspace/<path>
```

### Service Status
```bash
docker exec ai-ft-container ps aux | grep python
```

## Notes
- Always test changes in the Docker environment
- Check logs for errors: `docker logs ai-ft-container --tail 50`
- Monitor GPU if training: `nvidia-smi`
- Ensure web interface is accessible at http://localhost:8050