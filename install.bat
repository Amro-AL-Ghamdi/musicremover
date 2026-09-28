@echo off
rem One-step install for Windows: creates a private Python environment in server\.venv
rem and installs everything, with the PyTorch build that matches your GPU (NVIDIA, AMD,
rem or CPU-only if there's no usable GPU).
rem
rem   install.bat                 install
rem   install.bat --dry-run       show what would be installed
rem   install.bat --target cpu    force a build: cpu, nvidia or amd
setlocal
cd /d "%~dp0"

rem AMD's PyTorch for Windows needs Python 3.12, so prefer it when it's installed.
set "PY="
if defined PYTHON set "PY=%PYTHON%"
if not defined PY py -3.12 -c "import sys" >nul 2>nul && set "PY=py -3.12"
if not defined PY py -3 -c "import sys" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys" >nul 2>nul && set "PY=python"
if not defined PY (
  echo Python 3.9+ is needed. Install it from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" in the installer. Python 3.12 works for every GPU.
  pause
  exit /b 1
)
%PY% -c "import sys; sys.exit(sys.version_info < (3, 9))"
if errorlevel 1 (
  echo Python 3.9+ is needed. Install a newer one from https://www.python.org/downloads/
  pause
  exit /b 1
)

if not exist "server\.venv\Scripts\python.exe" (
  echo Creating the Python environment in server\.venv
  %PY% -m venv server\.venv
  if errorlevel 1 (
    echo Couldn't create the Python environment.
    pause
    exit /b 1
  )
)
server\.venv\Scripts\python.exe -m pip install --upgrade pip >nul
server\.venv\Scripts\python.exe server\install.py %*
if errorlevel 1 (
  echo.
  echo Install failed, see the messages above.
  pause
  exit /b 1
)

rem Start the server right away (not for --dry-run). Next time: double-click run.bat
echo %* | find "--dry-run" >nul && (pause & exit /b 0)
echo.
echo Starting the server (close this window to stop it; next time just double-click run.bat)
call "%~dp0run.bat"
