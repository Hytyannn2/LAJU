@echo off
rem Starts LAJU's GPS logger (engine\gps.py) in its own minimized window. It runs until that window is closed; at night it checks every 10 min.
rem The Windows task "LAJU GPS logger" runs this file every day at 06:00 (a second copy exits).
cd /d "%~dp0engine"
start "LAJU GPS logger" /min cmd /c ".venv\Scripts\python.exe gps.py || pause"
