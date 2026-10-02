#!/usr/bin/env bash
# Run every engine sequentially (GPU timings need the card to themselves). Logs → out/<engine>.log
# usage: bakeoff/run_all.sh [engine ...]   engines: qwen3-0.6B qwen3-1.7B kyutai chatterbox xtts
cd "$(dirname "$0")"
mkdir -p out
for e in "${@:-qwen3-0.6B qwen3-1.7B kyutai chatterbox xtts}"; do
  for x in $e; do
    case $x in
      qwen3-*) env QWEN_SIZE="${x#qwen3-}" qwen3/.venv/bin/python qwen3/run.py ;;
      *) "$x/.venv/bin/python" "$x/run.py" ;;
    esac > "out/$x.log" 2>&1
    echo "[$x] exit $? — $(grep -E 'synth_median_s|Error' "out/$x.log" | tail -1)"
  done
done
