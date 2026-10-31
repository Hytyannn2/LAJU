@echo off
rem Starts LAJU's GPS logger (engine\gps.py) in its own minimized window. Close that window to stop it.
cd /d "%~dp0engine"
start "LAJU GPS logger" /min cmd /k .venv\Scripts\python.exe gps.py
