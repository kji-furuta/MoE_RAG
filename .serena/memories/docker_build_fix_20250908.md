# Docker Build Fix - 2025-09-08

## Issue
Docker build failed after scripts directory cleanup because Dockerfile referenced deleted `scripts/setup_permissions.sh`

## Error Message
```
failed to solve: failed to compute cache key: failed to calculate checksum of ref a9jfex0q9x0u2vxitv06l8m6k::qh341mdg0r82grb6zn6yq9nz0: "/scripts/setup_permissions.sh": not found
```

## Fix Applied
Updated Dockerfile lines 136-137 and 147:
- Changed from: `scripts/setup_permissions.sh`
- Changed to: `scripts/setup/fix_permissions.sh`

The existing `fix_permissions.sh` script provides the same functionality (setting permissions for outputs directory).

## Build Command
```bash
cd docker
docker-compose build --no-cache ai-ft
```

## Note
The build process may take 10-20 minutes due to installing all dependencies including PyTorch, CUDA libraries, and NLP models.