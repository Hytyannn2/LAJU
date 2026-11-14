@echo off
rem Trains LAJU's AI for the night in its own minimized window, at low priority: 480 minutes the laptop is awake (or the
rem minutes given: Train AI.bat 240), 4 workers (2 if under 5 GB of memory is free). Run it again another night and it
rem continues from the saved brain. The first run builds its training maps first, about an hour more.
rem Then it designs the 64-bus network ("Phase 5: 64 buses", 5 min) and writes the exam to results\Phase 5 exam.txt.
rem Progress: engine\cache\train.log (or the website's AI training tab). Stop early: close the window; the best so far stays.
rem START IT BY DOUBLE-CLICK in File Explorer, or from a command line with: explorer.exe "<full path>\Train AI.bat"
rem Never from a terminal inside another app (Claude, VS Code): closing or restarting that app kills the training
rem (2026-11-09: a night's run and the GPS logger were lost that way).
rem While it trains Windows does not sleep on idle (closing the lid still does). Every 30 awake minutes the AI is also
rem saved to engine\models\checkpoints\ (the newest 5 stay). Each run leaves a record: results\exp-*.json.
rem After a change to the score (engine\sim.py SCORE_VERSION) the first run builds new training maps: about 1-2 hours
rem and 840 MB in engine\cache\ (older map sets stay there until you delete them).
cd /d "%~dp0engine"
if "%~2"=="run" goto run
set M=%~1
if "%M%"=="" set M=480
start "LAJU AI training" /min /belownormal cmd /c ""%~f0" %M% run"
exit /b

:run
set PYTHONUTF8=1
set W=4
for /f %%g in ('powershell -NoProfile -c "[int][math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)"') do if %%g LSS 5 set W=2
echo %date% %time%: %~1 minutes, %W% workers >> cache\train.log
.venv\Scripts\python.exe rl.py %~1 %W% >> cache\train.log 2>&1 || goto failed
.venv\Scripts\python.exe rl.py --design >> cache\train.log 2>&1 || goto failed
.venv\Scripts\python.exe exam.py "BAS.MY today" "Phase 3: 64 buses" "Phase 5: 64 buses" > "..\results\Phase 5 exam.txt" 2>&1 || goto failed
exit /b

:failed
echo Something stopped the run. See engine\cache\train.log and results\Phase 5 exam.txt
pause
