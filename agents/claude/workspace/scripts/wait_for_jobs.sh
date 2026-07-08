#!/bin/bash
cd /root/kaggle-vibe-template/agents/claude/workspace/scripts
a_done=0
b_done=0
while true; do
  if ! ps -p 102300 > /dev/null 2>&1; then
    if [ "$a_done" -eq 0 ]; then
      echo "exp024_done"
      tail -5 exp024_out.log
      a_done=1
    fi
  fi
  if ! ps -p 106438 > /dev/null 2>&1; then
    if [ "$b_done" -eq 0 ]; then
      echo "fit_full_exp025_done"
      tail -5 fit_full_exp025_out.log
      b_done=1
    fi
  fi
  if [ "$a_done" -eq 1 ] && [ "$b_done" -eq 1 ]; then
    echo "BOTH_DONE"
    break
  fi
  sleep 15
done
