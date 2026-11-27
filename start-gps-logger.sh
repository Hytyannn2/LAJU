#!/usr/bin/env bash
# Linux version of "Start GPS logger.bat": runs engine/gps.py in the background until stopped or the PC restarts.
#   ./start-gps-logger.sh          start (a second copy exits)
#   ./start-gps-logger.sh stop
# Output: engine/cache/gps.log. Data: engine/cache/gps/<date>.csv. No autostart.
cd "$(dirname "$(readlink -f "$0")")/engine" || exit 1

if [ "${1:-}" = stop ]; then
    pkill -f 'python -u gps\.py' && echo "GPS logger stopped." || echo "GPS logger was not running."
    exit
fi
if pgrep -f 'python -u gps\.py' >/dev/null; then
    echo "The GPS logger is already running."
    exit
fi
setsid -f .venv/bin/python -u gps.py </dev/null >>cache/gps.log 2>&1
echo "GPS logger started. Watch: tail -f \"$PWD/cache/gps.log\""
