@echo off
REM ---------------------------------------------------------------------------
REM VoiceGuard - one-command setup + run (Windows)
REM
REM   Double-click this file, or in cmd:  run.bat
REM   Different port:   set PORT=8080 && run.bat
REM   Skip the ML model: set VG_DETECTOR_MODE=heuristic && run.bat
REM
REM Creates .venv\, installs the CORE dependencies only, starts the server.
REM Plain cmd syntax only - no PowerShell, so double-click works.
REM ---------------------------------------------------------------------------
setlocal enabledelayedexpansion
cd /d "%~dp0"

if "%PORT%"=="" set PORT=8000
if "%HOST%"=="" set HOST=0.0.0.0
set VENV=.venv

echo.
echo ===============================================
echo   VoiceGuard - AI voice clone detector
echo ===============================================
echo.

REM ------------------------------------------------------------- 1. python --
set PY=
for %%C in ("py -3" "python" "python3") do (
    if not defined PY (
        %%~C -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
        if !errorlevel! equ 0 set PY=%%~C
    )
)

if not defined PY (
    echo [X] VoiceGuard needs Python 3.10 or newer, and it was not found.
    echo.
    echo     1. Download it from  https://www.python.org/downloads/
    echo     2. IMPORTANT: tick "Add python.exe to PATH" in the installer.
    echo     3. Close this window, open a new one, and run run.bat again.
    echo.
    pause
    exit /b 1
)

for /f "delims=" %%V in ('%PY% -V 2^>^&1') do echo [1/4] Using %%V

REM --------------------------------------------------------------- 2. venv --
if exist "%VENV%\Scripts\python.exe" (
    echo [2/4] Reusing existing virtual environment ^(%VENV%^)
) else (
    echo [2/4] Creating virtual environment in %VENV% ...
    %PY% -m venv "%VENV%"
    if errorlevel 1 (
        echo.
        echo [X] Could not create the virtual environment.
        echo     Try running this window as Administrator, or move the project
        echo     folder somewhere without special characters in the path.
        echo.
        pause
        exit /b 1
    )
)
set VPY=%VENV%\Scripts\python.exe

REM ------------------------------------------------------- 3. dependencies --
echo [3/4] Installing core dependencies ^(quick, only happens once^) ...
"%VPY%" -m pip install --upgrade pip --quiet
"%VPY%" -m pip install -r backend\requirements.txt --quiet
if errorlevel 1 (
    echo.
    echo [X] Dependency install failed.
    echo     Most likely no internet, or a company proxy blocking PyPI.
    echo     With a proxy:  "%VPY%" -m pip install --proxy http://user:pass@host:port -r backend\requirements.txt
    echo     Or use Docker: docker compose up
    echo.
    pause
    exit /b 1
)

"%VPY%" -c "import torch, transformers" >nul 2>&1
if errorlevel 1 (
    echo       ML deps not installed - running on the built-in heuristic detector.
    echo       Optional, to enable the pretrained ML model ^(~1 GB download^):
    echo         "%VPY%" -m pip install -r backend\requirements-ml.txt
) else (
    echo       ML deps detected - the pretrained deepfake model will be used.
)

REM -------------------------------------------------------------- 4. serve --
echo.
echo [4/4] Starting VoiceGuard - open  http://localhost:%PORT%
echo       Press Ctrl+C to stop.
echo.
"%VPY%" -m uvicorn main:app --app-dir backend --host %HOST% --port %PORT%

echo.
echo Server stopped.
pause
