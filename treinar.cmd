@echo off
setlocal
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Ambiente Python ausente. Consulte privado\INSTRUCOES.md.
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -X utf8 "%~dp0scripts\treinar.py" %*
exit /b %errorlevel%
