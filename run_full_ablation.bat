@echo off
setlocal enabledelayedexpansion

:: Set script directory as working directory
cd /d "%~dp0"

echo ==============================================================================
echo        CON-SOL-E 5.0 -- DINOv2 Ablation Study (100 Epochs Runner)
echo ==============================================================================
echo Working Directory: %CD%
echo Time: %DATE% %TIME%
echo.

:: ------------------------------------------------------------------------------
:: Step 1: Pull latest updates from remote GitHub repository
:: ------------------------------------------------------------------------------
echo [Step 1/5] Pulling latest updates from remote git repository...
git pull origin main
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Git pull failed or there are local conflicts. Continuing with local files...
)
echo.

:: ------------------------------------------------------------------------------
:: Step 2: Activate or create virtual environment
:: ------------------------------------------------------------------------------
echo [Step 2/5] Checking Python virtual environment (venv)...
if not exist "venv\Scripts\activate.bat" (
    echo Creating new virtual environment 'venv'...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment! Please ensure Python 3.10+ is installed.
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate venv!
    pause
    exit /b 1
)
echo Python Executable: 
where python
echo.

:: ------------------------------------------------------------------------------
:: Step 3: Install and update dependencies
:: ------------------------------------------------------------------------------
echo [Step 3/5] Updating and installing dependencies from requirements.txt...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Some dependencies had installation warnings. Proceeding...
)
echo.

:: ------------------------------------------------------------------------------
:: Step 4: Check environment credentials for MongoDB Atlas sync
:: ------------------------------------------------------------------------------
echo [Step 4/5] Checking environment configuration (.env)...
if not exist ".env" (
    if exist ".env.example" (
        echo [NOTICE] .env not found. Creating .env from .env.example template...
        copy .env.example .env
        echo [ACTION REQUIRED] Please edit .env with your MongoDB Atlas credentials if you want cloud telemetry sync.
    ) else (
        echo [WARNING] .env file not found. Cloud telemetry sync may run in offline mode.
    )
) else (
    echo [.env found - MongoDB Atlas real-time cloud sync enabled]
)
echo.

:: ------------------------------------------------------------------------------
:: Step 5: Run full ablation study (100 Epochs)
:: ------------------------------------------------------------------------------
echo [Step 5/5] Starting Full Ablation Study (100 Epochs)...
echo Output file: ablation_results_100ep.json
echo ------------------------------------------------------------------------------
python run_ablation.py --mode full --epochs 100 --output ablation_results_100ep.json

set RUN_EXIT_CODE=%ERRORLEVEL%

echo.
echo ==============================================================================
if %RUN_EXIT_CODE% EQU 0 (
    echo [SUCCESS] Full 100-Epoch Ablation Study Completed Successfully!
    echo Results saved to ablation_results_100ep.json and synced to MongoDB Atlas.
) else (
    echo [FAILED] Ablation Study exited with code %RUN_EXIT_CODE%.
    echo Check logs above for details. You can re-run this script anytime to resume.
)
echo ==============================================================================
echo.

pause
