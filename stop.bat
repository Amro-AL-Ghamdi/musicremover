@echo off
rem Stop the music-removal server started by run.bat / install.bat (or the Windows app),
rem including an install or update that is still running. The PowerShell part below
rem finds this folder's server.py / install.py (and a Music Remover server.py on the
rem server's port) and ends them with everything they started.
setlocal
set "MR_DIR=%~dp0"
if not defined MR_PORT set "MR_PORT=8765"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s = Get-Content -Raw -LiteralPath '%~f0'; Invoke-Expression ($s.Substring($s.IndexOf('#' + 'POWERSHELL' + '#') + 12))"
set "RESULT=%errorlevel%"
if "%~1"=="" timeout /t 3 >nul
exit /b %RESULT%

#POWERSHELL#
$dir = $env:MR_DIR.TrimEnd('\')
$paths = @("$dir\server\server.py", "$dir\server\install.py", "$dir\app\server\server.py", "$dir\app\server\install.py")
$ids = @()
foreach ($p in Get-CimInstance Win32_Process -Filter "Name like 'python%'") {
  $cmd = [string]$p.CommandLine
  foreach ($x in $paths) {
    if ($cmd.IndexOf($x, [StringComparison]::OrdinalIgnoreCase) -ge 0) { $ids += $p.ProcessId }
  }
}
# Whatever listens on the server's port, if it is a Music Remover server.py.
foreach ($c in (Get-NetTCPConnection -LocalPort ([int]$env:MR_PORT) -State Listen -ErrorAction SilentlyContinue)) {
  $p = Get-CimInstance Win32_Process -Filter "ProcessId=$($c.OwningProcess)"
  if ($p -and ([string]$p.CommandLine) -match 'server\.py') { $ids += $p.ProcessId }
}
$ids = @($ids | Sort-Object -Unique)
if ($ids.Count -eq 0) { "Music Remover isn't running."; exit 0 }
"Stopping Music Remover (processes: " + ($ids -join ", ") + ")"
foreach ($i in $ids) { taskkill /PID $i /T /F 2>$null | Out-Null }
"Stopped."
exit 0
