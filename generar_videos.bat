@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Primero ejecuta instalar.bat
 pause
 exit /b 1
)
rem Genera pantalla_mirada.mp4 y heatmap_sobre_pantalla.png para las tareas que aun no lo tengan.
rem Para regenerar todo: generar_videos.bat --forzar
.venv\Scripts\python.exe render.py %*
pause
