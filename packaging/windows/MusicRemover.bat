@echo off
rem Music Remover for Windows: starts the local server. PyTorch for your GPU and the
rem models are downloaded on the first start into %LOCALAPPDATA%\MusicRemover, along
rem with a copy of the extension.
setlocal
title Music Remover
set "MR_HOME=%LOCALAPPDATA%\MusicRemover"
set "MR_CACHE=%MR_HOME%\models"
rem PyTorch goes to the data folder (pip --user) and is found there at run time.
set "PYTHONUSERBASE=%MR_HOME%\python"
set "PIP_USER=1"
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE="
"%~dp0python\python.exe" "%~dp0app\server\launcher.py" %*
if errorlevel 1 (
  echo.
  echo Music Remover stopped with an error, see above.
  pause
)
