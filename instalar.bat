@echo off
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 goto error
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto error
.venv\Scripts\python.exe descargar_modelo.py
if errorlevel 1 goto error
echo Instalacion terminada. Abre iniciar.bat
pause
exit /b 0
:error
echo No se completo la instalacion. Conserva este mensaje para revisar el error.
pause
exit /b 1
