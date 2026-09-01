@echo off
REM ---------------------------------------------------------------------------
REM VoiceGuard - one-command setup + run (Windows)
REM
REM   Double-click this file, or in cmd:  run.bat
REM   Different port:      set PORT=8080 && run.bat
REM   Allow other devices: set HOST=0.0.0.0 && run.bat   (may prompt the firewall)
REM   Skip the ML model:   set VG_DETECTOR_MODE=heuristic && run.bat
REM
REM Creates .venv\, installs the CORE dependencies only, starts the server.
REM Plain cmd syntax only - no PowerShell - so double-click works.
REM
REM Deliberately avoids parenthesised ( ) blocks around any python -c "..."
REM probe: cmd mis-parses a ")" inside a quoted string inside a block, which is
REM a classic silent-failure in batch scripts.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

if "%PORT%"=="" set PORT=8000
REM 127.0.0.1 keeps Windows Defender Firewall quiet. Set HOST=0.0.0.0 yourself
REM if you want to reach the app from another device on your network.
if "%HOST%"=="" set HOST=127.0.0.1
set VENV=.venv
set VPY=%VENV%\Scripts\python.exe

echo.
echo ===============================================
echo   VoiceGuard - AI voice clone detector
echo ===============================================
echo.

REM ------------------------------------------------------------- 1. python --
REM "py -3" (the official Python launcher) is tried first because bare
REM "python"/"python3" on a machine without Python is a Microsoft Store stub
REM that pops open the Store instead of failing quietly.
set PY=
call :try_python py -3
if not defined PY call :try_python py
if not defined PY call :try_python python

if not defined PY goto :no_python

for /f "delims=" %%V in ('%PY% -V 2^>^&1') do echo [1/4] Using %%V

REM --------------------------------------------------------------- 2. venv --
call :venv_usable
if not errorlevel 1 goto :venv_ready

echo [2/4] Creating virtual environment in %VENV% ...
%PY% -m venv "%VENV%" >nul 2>&1
call :venv_usable
if not errorlevel 1 goto :venv_ready

REM Some Python installs ship a broken pip bootstrap. uv can build the
REM environment without it, so use that rather than dead-ending.
where uv >nul 2>&1
if errorlevel 1 goto :no_venv
echo       This Python's pip bootstrap is broken - falling back to 'uv' ...
if exist "%VENV%" rmdir /s /q "%VENV%"
uv venv "%VENV%"
if errorlevel 1 goto :no_venv
set USE_UV=1
goto :venv_ready

:venv_ready
if not defined USE_UV echo [2/4] Virtual environment ready ^(%VENV%^)

REM ------------------------------------------------------- 3. dependencies --
echo [3/4] Installing core dependencies ^(quick, only happens once^) ...
if defined USE_UV goto :install_uv

"%VPY%" -m pip install --upgrade pip --quiet
"%VPY%" -m pip install -r backend\requirements.txt --quiet
if errorlevel 1 goto :no_deps
goto :deps_done

:install_uv
uv pip install --python "%VPY%" -r backend\requirements.txt --quiet
if errorlevel 1 goto :no_deps

:deps_done
call :have_ml
if errorlevel 1 goto :no_ml_msg
echo       ML deps detected - the pretrained deepfake model will be used.
goto :serve

:no_ml_msg
echo       ML deps not installed - running on the built-in heuristic detector.
echo       This works. To enable the pretrained ML model ^(~1 GB download^):
echo         %VPY% -m pip install -r backend\requirements-ml.txt

REM -------------------------------------------------------------- 4. serve --
:serve
echo.
echo [4/4] Starting VoiceGuard - open  http://localhost:%PORT%
echo       Press Ctrl+C to stop.
echo.
"%VPY%" -m uvicorn main:app --app-dir backend --host %HOST% --port %PORT%

echo.
echo Server stopped.
pause
exit /b 0

REM ===========================================================================
REM Error paths
REM ===========================================================================
:no_python
echo [X] VoiceGuard needs Python 3.10 or newer, and it was not found.
echo.
echo     1. Download it from  https://www.python.org/downloads/
echo     2. IMPORTANT: tick "Add python.exe to PATH" in the installer.
echo     3. Close this window, open a NEW one, and run run.bat again.
echo.
pause
exit /b 1

:no_venv
echo.
echo [X] Could not create a working virtual environment.
echo     This is a problem with the Python install, not with VoiceGuard.
echo     Pick one:
echo       * Reinstall Python from python.org ^(tick "Add python.exe to PATH"^)
echo       * Install uv:  winget install astral-sh.uv    then re-run run.bat
echo       * Skip Python entirely:  docker compose up
echo     If the path to this folder has accents or "!" in it, move the folder.
echo.
pause
exit /b 1

:no_deps
echo.
echo [X] Dependency install failed.
echo     Most likely no internet, or a company proxy blocking PyPI.
echo     With a proxy:
echo       %VPY% -m pip install --proxy http://user:pass@host:port -r backend\requirements.txt
echo     Or use Docker: docker compose up
echo.
pause
exit /b 1

REM ===========================================================================
REM Subroutines - kept out of ( ) blocks on purpose, see the header note.
REM ===========================================================================

REM Probe one interpreter. %* is the whole command, e.g. "py -3".
:try_python
if defined PY goto :eof
%* -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if errorlevel 1 goto :eof
set PY=%*
goto :eof

REM errorlevel 0 = the venv exists AND its pip actually runs.
:venv_usable
if not exist "%VPY%" exit /b 1
if defined USE_UV exit /b 0
"%VPY%" -m pip --version >nul 2>&1
exit /b %errorlevel%

:have_ml
"%VPY%" -c "import torch, transformers" >nul 2>&1
exit /b %errorlevel%
