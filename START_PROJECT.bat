@echo off
setlocal

title HMS NeuroScope - EEG Seizure Detection Launcher
echo ===============================================================================
echo   HMS NeuroScope - Harmful Brain Activity Classifier and EEG Seizure Detection
echo   One-Click Windows Launcher
echo ===============================================================================
echo.

:: 1. Work from the folder containing this batch file
cd /d "%~dp0"
echo [1/6] Working directory: %CD%
echo.

:: 2. Find Python: prefer the "py" launcher, fall back to "python"
echo [2/6] Checking Python installation...
set "PY="
py -3 --version >nul 2>&1
if not errorlevel 1 set "PY=py -3"
if not defined PY (
    python --version >nul 2>&1
    if not errorlevel 1 set "PY=python"
)
if not defined PY goto :no_python
for /f "tokens=*" %%i in ('%PY% --version') do echo   Found %%i
echo   Python 3.11 or 3.12 is recommended.
echo.

:: 3. Create the virtual environment on first run
echo [3/6] Setting up virtual environment in .venv ...
if exist ".venv\Scripts\python.exe" goto :venv_ready
echo   Creating a fresh virtual environment, please wait...
%PY% -m venv .venv
if errorlevel 1 goto :venv_failed
echo   Virtual environment created.
:venv_ready
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :venv_failed
echo   Activated: %VIRTUAL_ENV%
echo.

:: 4. Install dependencies if anything is missing
echo [4/6] Verifying Python dependencies...
python -c "import mne, fastapi, uvicorn, multipart, sklearn, scipy, plotly, joblib, matplotlib, requests, pytest, numpy, pandas, wfdb" >nul 2>&1
if not errorlevel 1 goto :deps_ready
echo   Installing packages from requirements.txt
echo   The first run can take 3 to 5 minutes, please wait...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto :deps_failed
echo   Dependencies installed.
:deps_ready
echo   All dependencies are present.
echo.

:: 5. Check model and sample data
echo [5/6] Checking model and sample EEG recordings...
if exist "results\trained_model.joblib" goto :model_ready
echo   [WARNING] results\trained_model.joblib is missing - training it now...
python scripts\run_real_experiment.py
if errorlevel 1 goto :train_failed
:model_ready
echo   Found trained model: results\trained_model.joblib
if exist "data\raw\chb01\chb01_03.edf" (
    echo   Found sample CHB-MIT recordings in data\raw
) else (
    echo   [NOTICE] data\raw\chb01\chb01_03.edf not found - see DATA_SETUP.md
)
echo.

:: 6. Launch the dashboard
echo [6/6] Launching the dashboard...
echo.
echo ===============================================================================
echo   Opening in your browser at: http://localhost:8501
echo   To STOP the server, close this window or press Ctrl + C.
echo ===============================================================================
echo.
python run_app.py --port 8501
if errorlevel 1 (
    echo.
    echo [ERROR] The web server stopped with an error.
    echo If port 8501 is busy, run:  .venv\Scripts\python run_app.py --port 8502
)
pause
exit /b 0

:no_python
echo.
echo [ERROR] Python was not found.
echo.
echo Install Python 3.11 or 3.12 from https://www.python.org/downloads/windows/
echo and tick "Add python.exe to PATH" on the first installer screen.
echo Then run this file again.
echo.
pause
exit /b 1

:venv_failed
echo.
echo [ERROR] Could not create or activate the virtual environment in .venv
echo Make sure this folder is not read-only, for example extract the zip first
echo instead of running from inside it, then try again.
echo.
pause
exit /b 1

:deps_failed
echo.
echo [ERROR] Installing requirements failed. Check your internet connection
echo and the messages above, then run this file again.
echo.
pause
exit /b 1

:train_failed
echo.
echo [ERROR] Automatic model training failed. See the messages above.
echo.
pause
exit /b 1
