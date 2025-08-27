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

# Navigate to docker directory
cd docker

# Stop all containers
print_info "Stopping Docker containers..."
docker-compose down

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
echo "To restart the environment, run: ./start_dev_env.sh"
echo "To clean all data and volumes, run: cd docker && docker-compose down -v"