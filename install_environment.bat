@echo off
REM LLM Format Benchmarker - Environment Setup Script for Windows
REM This script sets up the Python environment and installs all required dependencies

echo.
echo ================================================
echo LLM Format Benchmarker - Environment Setup
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Checking Python version...
python -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"

REM Check Python version is 3.8+
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python 3.8+ is required
    echo Current Python version is too old
    pause
    exit /b 1
)

echo Python version check passed!
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
) else (
    echo Virtual environment already exists
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo Virtual environment activated!
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo WARNING: Failed to upgrade pip, continuing anyway...
)

echo.
echo Installing PyTorch with CPU support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo.
echo Installing main dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements
    echo Make sure requirements.txt exists in the current directory
    pause
    exit /b 1
)

echo.
echo Installing Intel optimizations (optional)...
pip install intel-extension-for-pytorch
if %errorlevel% neq 0 (
    echo WARNING: Intel Extension for PyTorch installation failed
    echo This is optional and won't affect basic functionality
)

echo.
echo Installing OpenVINO (optional)...
pip install openvino openvino-genai
if %errorlevel% neq 0 (
    echo WARNING: OpenVINO installation failed
    echo This is optional but recommended for Intel hardware optimization
)

echo.
echo ================================================
echo Environment Setup Complete!
echo ================================================
echo.
echo To activate the environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
echo To test the installation, run:
echo   python run_benchmark.py --detect-hardware
echo.
echo To start with interactive examples:
echo   python examples\run_examples.py
echo.
pause
