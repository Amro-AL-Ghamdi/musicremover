@echo off
rem Start the music-removal server (after install.bat). Close this window to stop it.
cd /d "%~dp0server"
if not exist ".venv\Scripts\python.exe" (
  echo Not installed yet: run install.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe server.py %*
pause
