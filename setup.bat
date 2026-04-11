@echo off
REM Setup script for Sox2 expression prediction project (Windows)
REM Detects if conda environment exists, creates if needed

setlocal enabledelayedexpansion

echo.
echo ==========================================
echo Sox2 Expression Prediction - Setup Script
echo ==========================================
echo.

REM Check if conda is installed
where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: conda is not installed or not in PATH
    echo Please install Anaconda or Miniconda from: https://www.anaconda.com/download
    pause
    exit /b 1
)

echo. ✓ conda found
echo.

REM Check if environment already exists
set ENV_NAME=sox2-alphagenome
conda env list | findstr /I "^%ENV_NAME% " >nul 2>nul

if errorlevel 1 (
    REM Environment does not exist, create it
    echo Creating conda environment from environment.yml...
    echo.
    call conda env create -f environment.yml --yes
    
    echo.
    echo Installing pip packages...
    call conda run -n %ENV_NAME% pip install --upgrade pip setuptools wheel
    call conda run -n %ENV_NAME% pip install huggingface-hub pyfaidx
    call conda run -n %ENV_NAME% pip install alphagenome-pytorch==0.2.8
    
    echo.
    echo ==========================================
    echo Environment created successfully!
    echo ==========================================
    echo.
    echo To activate the environment, run:
    echo   conda activate %ENV_NAME%
    echo.
    echo To verify installation, run:
    echo   conda activate %ENV_NAME%
    echo   python example_inference.py
    echo.
    echo To deactivate the environment when done, run:
    echo   conda deactivate
    echo.
) else (
    REM Environment already exists
    echo. ✓ Environment '%ENV_NAME%' already exists
    echo.
    echo ==========================================
    echo Environment already installed!
    echo ==========================================
    echo.
    echo To activate the environment, run:
    echo   conda activate %ENV_NAME%
    echo.
    echo To verify installation, run:
    echo   conda activate %ENV_NAME%
    echo   python example_inference.py
    echo.
)

pause
