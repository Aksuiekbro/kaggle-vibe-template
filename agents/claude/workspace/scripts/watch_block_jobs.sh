#!/bin/bash
cd /root/kaggle-vibe-template/agents/claude/workspace || exit 1

prev056=0
prev058=0
prev061=0

while true; do
  n056=$(wc -l < fit_full_exp056_out.log)
  if [ "$n056" != "$prev056" ]; then
    tail -n +$((prev056+1)) fit_full_exp056_out.log
    prev056=$n056
  fi

  n058=$(wc -l < fit_full_exp058_out.log)
  if [ "$n058" != "$prev058" ]; then
    tail -n +$((prev058+1)) fit_full_exp058_out.log
    prev058=$n058
  fi

  n061=$(wc -l < exp061_probe.log)
  if [ "$n061" != "$prev061" ]; then
    tail -n +$((prev061+1)) exp061_probe.log
    prev061=$n061
  fi

  alive056=$(ps -p 183552 -o pid= | wc -l)
  alive058=$(ps -p 184321 -o pid= | wc -l)
  alive061=$(ps -p 184706 -o pid= | wc -l)

  if [ "$alive056" = "0" ] && [ "$alive058" = "0" ] && [ "$alive061" = "0" ]; then
    echo "ALL_DONE"
    break
  fi

  sleep 20
done
