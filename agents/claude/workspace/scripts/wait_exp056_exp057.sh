#!/bin/bash
cd /root/kaggle-vibe-template/agents/claude/workspace
d56=0
d57=0
while [ "$d56" -eq 0 ] || [ "$d57" -eq 0 ]; do
  if [ "$d56" -eq 0 ]; then
    if grep -qE "total time:|Traceback|Error" exp056_probe.log 2>/dev/null; then
      echo "=== exp056 done ==="
      tail -n 30 exp056_probe.log
      d56=1
    elif ! pgrep -f probe_exp056_iterative_pseudolabel.py >/dev/null; then
      echo "=== exp056 process gone, no marker ==="
      tail -n 30 exp056_probe.log
      d56=1
    fi
  fi
  if [ "$d57" -eq 0 ]; then
    if grep -qE "total time:|Traceback|Error" exp057_probe.log 2>/dev/null; then
      echo "=== exp057 done ==="
      tail -n 30 exp057_probe.log
      d57=1
    elif ! pgrep -f probe_exp057_threshold_finetune.py >/dev/null; then
      echo "=== exp057 process gone, no marker ==="
      tail -n 30 exp057_probe.log
      d57=1
    fi
  fi
  sleep 15
done
echo "BOTH DONE"
