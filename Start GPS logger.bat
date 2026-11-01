@echo off
rem Starts LAJU's GPS logger (engine\gps.py) in its own minimized window. It closes by itself at 23:00 (gps.STOP_HOUR);
rem close that window to stop it sooner. The Windows task "LAJU GPS logger" runs this file every day at 06:00.
cd /d "%~dp0engine"
start "LAJU GPS logger" /min cmd /c ".venv\Scripts\python.exe gps.py || pause"
