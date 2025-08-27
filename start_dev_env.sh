#!/bin/bash

# Development Environment Startup Script for MoE_RAG
# This script starts the complete development environment

set -e

echo "========================================="
echo "MoE_RAG Development Environment Startup"
echo "========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if running in WSL2
if grep -qi microsoft /proc/version; then
    print_success "Running in WSL2 environment"
else
    print_info "Not running in WSL2 - some features may behave differently"
fi

# Navigate to docker directory
cd docker

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

print_success "Docker is running"

# Check for .env file
if [ ! -f ".env" ]; then
    print_info "Creating .env file from example..."
    cp .env.example .env
    print_info "Please update docker/.env with your API keys before continuing"
    read -p "Press Enter when ready to continue..."
fi

# Build and start containers
print_info "Building Docker containers..."
docker-compose build --no-cache

print_info "Starting Docker containers..."
docker-compose up -d

# Wait for containers to be ready
print_info "Waiting for containers to be ready..."
sleep 10

# Check container status
if docker ps | grep -q ai-ft-container; then
    print_success "Main container is running"
else
    print_error "Main container failed to start"
    docker logs ai-ft-container --tail 50
    exit 1
fi

if docker ps | grep -q ai-ft-qdrant; then
    print_success "Qdrant container is running"
else
    print_error "Qdrant container failed to start"
    docker logs ai-ft-qdrant --tail 50
    exit 1
fi

# Start the web interface
print_info "Starting web interface..."
docker exec ai-ft-container bash -c "cd /workspace && python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload" &

# Wait for the service to start
print_info "Waiting for services to start..."
sleep 10

# Check service health
print_info "Checking service health..."
if curl -s http://localhost:8050/rag/health > /dev/null; then
    print_success "RAG service is healthy"
else
    print_error "RAG service health check failed"
fi

# Display access information
echo ""
echo "========================================="
echo "Development Environment Ready!"
echo "========================================="
echo ""
echo "Access points:"
echo "  - Web Interface: http://localhost:8050"
echo "  - RAG API: http://localhost:8050/rag"
echo "  - Fine-tuning API: http://localhost:8050/api"
echo "  - Qdrant UI: http://localhost:6333/dashboard"
echo "  - Jupyter Lab: http://localhost:8888"
echo "  - TensorBoard: http://localhost:6006"
echo ""
echo "Useful commands:"
echo "  - View logs: docker logs -f ai-ft-container"
echo "  - Enter container: docker exec -it ai-ft-container bash"
echo "  - Stop environment: docker-compose down"
echo "  - Clean restart: docker-compose down && docker-compose up -d --build"
echo ""
echo "To test the RAG system:"
echo "  curl -X POST http://localhost:8050/rag/query \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"query\": \"設計速度80km/hの道路の最小曲線半径は？\", \"top_k\": 5}'"
echo ""
print_success "Development environment is ready!"