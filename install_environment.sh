#!/bin/bash

# LLM Format Benchmarker - Environment Setup Script for Linux/Mac
# This script sets up the Python environment and installs all required dependencies

set -e  # Exit on any error

echo ""
echo "================================================"
echo "LLM Format Benchmarker - Environment Setup"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    echo "Please install Python 3.8+ using your system's package manager:"
    echo ""
    echo "Ubuntu/Debian: sudo apt update && sudo apt install python3 python3-pip python3-venv"
    echo "CentOS/RHEL:   sudo yum install python3 python3-pip"
    echo "macOS:         brew install python3"
    echo ""
    exit 1
fi

print_status "Checking Python version..."
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "Python $PYTHON_VERSION"

# Check Python version is 3.8+
python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null
if [ $? -ne 0 ]; then
    print_error "Python 3.8+ is required"
    echo "Current Python version is too old"
    exit 1
fi

print_status "Python version check passed!"
echo ""

# Check if virtual environment already exists
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        print_error "Failed to create virtual environment"
        echo "Make sure python3-venv is installed:"
        echo "Ubuntu/Debian: sudo apt install python3-venv"
        exit 1
    fi
    print_status "Virtual environment created successfully!"
else
    print_status "Virtual environment already exists"
fi

echo ""
print_status "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    print_error "Failed to activate virtual environment"
    exit 1
fi

print_status "Virtual environment activated!"
echo ""

# Upgrade pip
print_status "Upgrading pip..."
python -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    print_warning "Failed to upgrade pip, continuing anyway..."
fi

echo ""
print_header "Installing PyTorch with CPU support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if [ $? -ne 0 ]; then
    print_error "Failed to install PyTorch"
    exit 1
fi

echo ""
print_header "Installing main dependencies from requirements.txt..."
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found in current directory"
    exit 1
fi

pip install -r requirements.txt
if [ $? -ne 0 ]; then
    print_error "Failed to install requirements"
    exit 1
fi

echo ""
print_header "Installing Intel optimizations (optional)..."
pip install intel-extension-for-pytorch
if [ $? -ne 0 ]; then
    print_warning "Intel Extension for PyTorch installation failed"
    print_warning "This is optional and won't affect basic functionality"
fi

echo ""
print_header "Installing OpenVINO (optional)..."
pip install openvino openvino-genai
if [ $? -ne 0 ]; then
    print_warning "OpenVINO installation failed"
    print_warning "This is optional but recommended for Intel hardware optimization"
fi

echo ""
echo "================================================"
print_status "Environment Setup Complete!"
echo "================================================"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "  python run_benchmark.py --detect-hardware"
echo ""
echo "To start with interactive examples:"
echo "  python examples/run_examples.py"
echo ""

# Check if we're on macOS and warn about potential Intel optimizations
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    print_warning "Note for macOS users:"
    echo "Intel hardware optimizations may have limited support on macOS."
    echo "For best performance, consider using Linux on Intel hardware."
fi

print_status "Setup completed successfully! 🎉"
