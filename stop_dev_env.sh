#!/bin/bash

# Development Environment Shutdown Script for MoE_RAG
# This script gracefully stops the development environment

set -e

echo "========================================="
echo "MoE_RAG Development Environment Shutdown"
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

# Resolve repository root irrespective of where the script is invoked
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

# Determine docker compose command (plugin or standalone)
if docker compose version > /dev/null 2>&1; then
    DOCKER_COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE=(docker-compose)
else
    print_error "Neither 'docker compose' nor 'docker-compose' is available. Install Docker Desktop or the Docker CLI plugin."
    exit 1
fi

# Navigate to docker directory
cd "$REPO_ROOT/docker"

# Stop all containers
print_info "Stopping Docker containers..."
"${DOCKER_COMPOSE[@]}" down

print_success "All containers stopped"

# Optional: Clean up volumes (commented out by default to preserve data)
# print_info "Cleaning up volumes..."
# docker-compose down -v
# print_success "Volumes cleaned"

echo ""
echo "========================================="
echo "Development Environment Stopped"
echo "========================================="
echo ""
COMPOSE_DISPLAY="${DOCKER_COMPOSE[*]}"
echo "To restart the environment, run: ./start_dev_env.sh"
echo "To clean all data and volumes, run: cd docker && ${COMPOSE_DISPLAY} down -v"

