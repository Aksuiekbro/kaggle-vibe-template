#!/bin/bash
a_done=0
b_done=0
while true; do
  if ! ps -p 113436 > /dev/null 2>&1; then
    if [ "$a_done" -eq 0 ]; then
      echo "exp033_done"
      tail -20 exp033.log
      a_done=1
    fi
  fi
  if ! ps -p 115415 > /dev/null 2>&1; then
    if [ "$b_done" -eq 0 ]; then
      echo "exp035_full_done"
      tail -20 exp035_full.log
      b_done=1
    fi
  fi
  if [ "$a_done" -eq 1 ] && [ "$b_done" -eq 1 ]; then
    echo "BOTH_DONE"
    break
  fi
  sleep 20
done
