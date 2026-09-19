@echo off
setlocal
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Ambiente Python ausente. Consulte privado\INSTRUCOES.md.
  exit /b 1
)
if "%~1"=="" (
  echo Uso: .\testar_microfone.cmd COM3
  echo Substitua COM3 pela porta da sua placa. Portas encontradas:
  "%~dp0.venv\Scripts\python.exe" -m serial.tools.list_ports
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -X utf8 "%~dp0scripts\build_firmware.py" --environment mic_test --upload --port "%~1"
if errorlevel 1 exit /b 1
echo Teste instalado. Fique em silencio e depois fale perto do microfone. Ctrl+C encerra o monitor.
"%~dp0.venv\Scripts\python.exe" -m platformio device monitor -b 115200 -p "%~1"
exit /b %errorlevel%
