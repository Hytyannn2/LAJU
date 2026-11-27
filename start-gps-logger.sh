#!/usr/bin/env bash
# Linux version of "Start GPS logger.bat": runs engine/gps.py in the background until stopped or the PC restarts.
#   ./start-gps-logger.sh          start (a second copy exits); in a terminal it then shows the log live
#   ./start-gps-logger.sh stop
# Double-click: "Run in Terminal" shows the log; closing that window leaves the logger running. "Run" starts it silently.
# Output: engine/cache/gps.log. Data: engine/cache/gps/<date>.csv. No autostart.
cd "$(dirname "$(readlink -f "$0")")/engine" || exit 1

if [ "${1:-}" = stop ]; then
    pkill -f 'python -u gps\.py' && echo "GPS logger stopped." || echo "GPS logger was not running."
    exit
fi
if pgrep -f 'python -u gps\.py' >/dev/null; then
    echo "The GPS logger is already running."
else
    setsid -f .venv/bin/python -u gps.py </dev/null >>cache/gps.log 2>&1
    echo "GPS logger started."
fi
if [ -t 1 ]; then
    echo "Live log below. Closing this window does not stop the logger (stop: ./start-gps-logger.sh stop)."
    exec tail -n 5 -f cache/gps.log
fi
echo "Watch: tail -f \"$PWD/cache/gps.log\""
