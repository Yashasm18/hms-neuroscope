@echo off
setlocal

title HMS NeuroScope - Automated Test Suite
echo ===============================================================================
echo   HMS NeuroScope - Automated PyTest Verification
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
echo Running the unit tests: signal filtering, windowing, feature extraction,
echo patient-level split isolation and metric computations...
echo.

python -m pytest -v
if errorlevel 1 (
    echo.
    echo ===============================================================================
    echo   [FAIL] One or more tests failed. Please inspect the output above.
    echo ===============================================================================
) else (
    echo.
    echo ===============================================================================
    echo   [SUCCESS] All unit tests passed.
    echo ===============================================================================
)

echo.
pause
