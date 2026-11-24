@echo off
setlocal

title HMS NeuroScope - Real CHB-MIT Experiment Pipeline
echo ===============================================================================
echo   HMS NeuroScope - Real Experiment Runner
echo ===============================================================================
echo.

cd /d "%~dp0"
echo Working directory: %CD%
echo.

if exist ".venv\Scripts\activate.bat" (
    echo Activating the local virtual environment in .venv
    call ".venv\Scripts\activate.bat"
) else (
    echo [NOTICE] .venv not found - run START_PROJECT.bat once first.
    echo Trying with the system Python instead...
)

echo.
echo WARNING: This re-extracts multichannel EEG features, re-trains the
echo Random Forest model, calibrates the validation threshold and overwrites
echo the artifacts in the results folder.
echo.
echo Estimated runtime: about 15 to 25 seconds.
echo.
set "CONFIRM=Y"
set /p CONFIRM="Do you want to proceed? (Y/N) [default: Y]: "
if /i "%CONFIRM%"=="N" (
    echo.
    echo Operation cancelled.
    pause
    exit /b 0
)

echo.
echo [1/2] Auditing single-recording split timestamps...
python scripts\audit_chb01_03.py

echo.
echo [2/2] Running the patient-level holdout experiment on CHB-MIT...
python scripts\run_real_experiment.py
if errorlevel 1 (
    echo.
    echo ===============================================================================
    echo   [FAIL] Experiment failed. Please check the error trace above.
    echo ===============================================================================
) else (
    echo.
    echo ===============================================================================
    echo   [SUCCESS] Experiment completed. Artifacts updated in the results folder.
    echo ===============================================================================
)

echo.
pause
