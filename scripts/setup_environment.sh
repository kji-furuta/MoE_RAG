#!/bin/bash

set -e

echo "=€ AI Fine-Tuning Environment Setup Script"
echo "=========================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if CUDA is available
check_cuda() {
    print_status "Checking CUDA availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
        print_success "NVIDIA driver found: $CUDA_VERSION"
        
        # Get CUDA toolkit version if available
        if command -v nvcc &> /dev/null; then
            CUDA_TOOLKIT_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
            print_success "CUDA toolkit version: $CUDA_TOOLKIT_VERSION"
        else
            print_warning "CUDA toolkit not found. PyTorch will use CPU or download CUDA runtime."
        fi
        
        # Check GPU memory
        print_status "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
        
        return 0
    else
        print_warning "NVIDIA GPU not detected. Will proceed with CPU-only setup."
        return 1
    fi
}

# Check PyTorch CUDA compatibility
check_pytorch_cuda() {
    print_status "Checking PyTorch CUDA compatibility..."
    
    if python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null; then
        print_success "PyTorch CUDA compatibility check passed"
        
        # Check GPU count
        GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
        if [ "$GPU_COUNT" -gt 0 ]; then
            print_success "Detected $GPU_COUNT GPU(s) available for training"
            
            # List available GPUs
            for i in $(seq 0 $((GPU_COUNT-1))); do
                GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name($i))" 2>/dev/null || echo "Unknown")
                print_status "GPU $i: $GPU_NAME"
            done
        else
            print_warning "No GPUs detected by PyTorch"
        fi
    else
        print_error "PyTorch not installed or not working properly"
        return 1
    fi
}

# Install uv if not available
install_uv() {
    print_status "Checking for uv package manager..."
    
    if ! command -v uv &> /dev/null; then
        print_status "Installing uv package manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
        
        if command -v uv &> /dev/null; then
            print_success "uv installed successfully"
        else
            print_error "Failed to install uv"
            return 1
        fi
    else
        print_success "uv is already installed"
    fi
}

# Setup Python environment with uv
setup_python_env() {
    print_status "Setting up Python environment with uv..."
    
    # Check if we're in a virtual environment
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        print_status "Already in virtual environment: $VIRTUAL_ENV"
    else
        # Create virtual environment if it doesn't exist
        if [ ! -d ".venv" ]; then
            print_status "Creating virtual environment..."
            uv venv .venv
        fi
        
        print_status "Activating virtual environment..."
        source .venv/bin/activate
    fi
    
    print_success "Python environment ready"
}

# Install packages with uv
install_packages() {
    print_status "Installing packages with uv..."
    
    # Install packages from requirements.txt
    if [ -f "requirements.txt" ]; then
        print_status "Installing from requirements.txt..."
        uv pip install -r requirements.txt
        print_success "Requirements installed successfully"
    else
        print_error "requirements.txt not found"
        return 1
    fi
    
    # Install additional development packages
    print_status "Installing development packages..."
    uv pip install jupyter ipykernel
    
    # Setup Jupyter kernel
    python -m ipykernel install --user --name ai-ft --display-name "AI Fine-Tuning"
    
    print_success "All packages installed successfully"
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Check key packages
    python3 -c "
import sys
packages = ['torch', 'transformers', 'datasets', 'peft', 'accelerate']
for pkg in packages:
    try:
        __import__(pkg)
        print(f' {pkg} imported successfully')
    except ImportError as e:
        print(f' {pkg} failed to import: {e}')
        sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "All key packages verified successfully"
    else
        print_error "Package verification failed"
        return 1
    fi
}

# Check multi-GPU setup
check_multi_gpu() {
    print_status "Checking multi-GPU setup..."
    
    python3 -c "
import torch
import torch.distributed as dist

gpu_count = torch.cuda.device_count()
print(f'Available GPUs: {gpu_count}')

if gpu_count > 1:
    print('Multi-GPU training capabilities:')
    print(f'- DataParallel: Available')
    print(f'- DistributedDataParallel: Available')
    
    # Check NCCL availability for multi-GPU training
    try:
        if torch.distributed.is_nccl_available():
            print('- NCCL backend: Available')
        else:
            print('- NCCL backend: Not available')
    except:
        print('- NCCL backend: Not available')
        
    # Memory info for each GPU
    for i in range(gpu_count):
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'- GPU {i}: {total_memory:.1f} GB memory')
else:
    print('Single GPU or CPU training mode')
"
}

# Main execution
main() {
    echo ""
    print_status "Starting environment setup..."
    
    # Change to project directory
    cd "$(dirname "$0")/.."
    
    # Run setup steps
    check_cuda
    CUDA_AVAILABLE=$?
    
    install_uv || exit 1
    setup_python_env || exit 1
    install_packages || exit 1
    verify_installation || exit 1
    
    if [ $CUDA_AVAILABLE -eq 0 ]; then
        check_pytorch_cuda || exit 1
        check_multi_gpu
    fi
    
    echo ""
    print_success "Environment setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the environment: source .venv/bin/activate"
    echo "2. Configure your training settings in config/"
    echo "3. Run training: python scripts/train_model.py"
    echo ""
}

# Run main function
main "$@"