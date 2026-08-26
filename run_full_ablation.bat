@echo off
setlocal enabledelayedexpansion

:: Set script directory as working directory
cd /d "%~dp0"

echo ==============================================================================
echo        CON-SOL-E 5.0 -- DINOv2 Ablation Study (Full Runner)
echo ==============================================================================
echo Working Directory: %CD%
echo Time: %DATE% %TIME%
echo.

:: Tunables. Override from the command line, e.g.:
::   run_full_ablation.bat 100 6
set EPOCHS=%1
if "%EPOCHS%"=="" set EPOCHS=100
set WORKERS=%2
if "%WORKERS%"=="" set WORKERS=6

echo Configuration: %EPOCHS% epochs, %WORKERS% dataloader workers
echo.

:: ------------------------------------------------------------------------------
:: Step 1: Pull latest updates from remote GitHub repository
:: ------------------------------------------------------------------------------
echo [Step 1/6] Pulling latest updates from remote git repository...
git pull origin main
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Git pull failed or there are local conflicts. Continuing with local files...
)
echo.

:: ------------------------------------------------------------------------------
:: Step 2: Activate or create virtual environment
:: ------------------------------------------------------------------------------
echo [Step 2/6] Checking Python virtual environment (venv)...
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
:: Step 3: Install dependencies (CUDA build of PyTorch first)
:: ------------------------------------------------------------------------------
echo [Step 3/6] Updating and installing dependencies...
python -m pip install --upgrade pip

:: Install torch from the CUDA 12.4 index explicitly. The default PyPI wheel
:: can resolve to a CPU-only build, which silently turns a 6-hour study into a
:: multi-day one. cu124 wheels run fine on the 560.94 / CUDA 12.6 driver.
echo Installing CUDA-enabled PyTorch (cu124)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] CUDA PyTorch install had issues. Falling back to requirements.txt resolution.
)

pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Some dependencies had installation warnings. Proceeding...
)
echo.

:: ------------------------------------------------------------------------------
:: Step 4: Verify CUDA is actually visible to PyTorch
:: ------------------------------------------------------------------------------
echo [Step 4/6] Verifying GPU availability...
python -c "import torch; ok=torch.cuda.is_available(); print('CUDA available:', ok); print('Device:', torch.cuda.get_device_name(0) if ok else 'CPU only'); raise SystemExit(0 if ok else 1)"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] PyTorch cannot see a CUDA GPU. The study will run on CPU and take days.
    echo           Fix with: pip uninstall -y torch torchvision ^&^& pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    echo.
    choice /C YN /M "Continue on CPU anyway"
    if !ERRORLEVEL! EQU 2 exit /b 1
)
echo.

:: ------------------------------------------------------------------------------
:: Step 5: Check environment credentials for MongoDB Atlas sync
:: ------------------------------------------------------------------------------
echo [Step 5/6] Checking environment configuration (.env)...
if not exist ".env" (
    echo [NOTICE] .env not found.
    if exist ".env.example" (
        copy .env.example .env >nul
        echo [ACTION REQUIRED] A template .env was created. Edit it with your MongoDB Atlas
        echo                   credentials, or the run will simply queue results offline
        echo                   and upload them automatically once .env is filled in.
    )
) else (
    echo [.env found - MongoDB Atlas real-time cloud sync enabled]
)

:: Push anything left in the offline queue from a previous run.
python upload_to_cloud.py --flush
echo.

:: ------------------------------------------------------------------------------
:: Step 6: Run the ablation study
:: ------------------------------------------------------------------------------
echo [Step 6/6] Starting Ablation Study (%EPOCHS% epochs)...
echo Output file: ablation_results.json
echo Each variant runs in its own subprocess, so one crash cannot stop the study.
echo ------------------------------------------------------------------------------
python run_ablation.py --mode full --epochs %EPOCHS% --num-workers %WORKERS%

set RUN_EXIT_CODE=%ERRORLEVEL%

echo.
echo Flushing any results queued while offline...
python upload_to_cloud.py --flush

echo.
echo ==============================================================================
if %RUN_EXIT_CODE% EQU 0 (
    echo [SUCCESS] Ablation Study Completed!
    echo Results saved to ablation_results.json and synced to MongoDB Atlas.
) else (
    echo [FAILED] Ablation Study exited with code %RUN_EXIT_CODE%.
    echo Check logs above. Re-run this script anytime to resume from the last epoch.
)
echo ==============================================================================
echo.

pause
