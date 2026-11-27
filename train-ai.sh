#!/usr/bin/env bash
# Linux version of "Train AI.bat" (what a run does and writes is described there).
#   ./train-ai.sh          train for 480 minutes in the background, then design the network and write the exam
#   ./train-ai.sh 240      the same for 240 minutes
#   ./train-ai.sh stop     stop early; the best so far stays
# Progress: engine/cache/train.log (or the website's AI training tab). Closing the terminal does not stop the run.
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/engine" || exit 1
M=${1:-480}

if [ "$M" = stop ]; then
    pkill -f 'train-ai\.sh [0-9.]+ run'  # the steps first, so the next one does not start
    pkill -f 'python -u (rl|exam)\.py' && echo "Training stopped." || echo "No training was running."
    exit
fi
[[ $M =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "Usage: $0 [minutes | stop]"; exit 1; }

if [ "${2:-}" != run ]; then
    flock -n "$SELF" true || { echo "The AI is already training."; exit 1; }
    # own session, low priority, no idle sleep while it runs
    setsid -f nice -n 10 systemd-inhibit --what=idle:sleep --who=LAJU --why="AI training" \
        "$SELF" "$M" run </dev/null >>cache/train.log 2>&1
    echo "Training for $M minutes in the background. Watch: tail -f \"$PWD/cache/train.log\""
    exit
fi

exec 9<"$SELF"
flock -n 9 || exit 1  # held until the last step ends: one run at a time
W=$(awk '/^MemAvailable/ {print ($2 < 5 * 1048576) ? 2 : 4}' /proc/meminfo)  # 2 workers under 5 GB free
echo "$(date): $M minutes, $W workers"
.venv/bin/python -u rl.py "$M" "$W" &&
    .venv/bin/python -u rl.py --design &&
    .venv/bin/python -u exam.py "BAS.MY today" "Phase 3: 64 buses" "Phase 5: 64 buses" >"../results/Phase 5 exam.txt" 2>&1 ||
    echo "Something stopped the run. See engine/cache/train.log and results/Phase 5 exam.txt"
