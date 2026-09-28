@echo off
rem Start the music-removal server (after install.bat). Close this window to stop it.
rem If an update changed the requirements, the installer runs first.
cd /d "%~dp0server"
if not exist ".venv\Scripts\python.exe" (
  echo Not installed yet: run install.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe launcher.py %*
pause
