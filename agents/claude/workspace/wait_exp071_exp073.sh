#!/bin/bash
EXP071_DONE=0
EXP073_DONE=0
while [ "$EXP071_DONE" -eq 0 ] || [ "$EXP073_DONE" -eq 0 ]; do
  if [ "$EXP071_DONE" -eq 0 ] && ! ps -p 214271 > /dev/null 2>&1; then
    echo "=== exp071 full CV process exited ==="
    tail -20 full_cv_exp071.log
    EXP071_DONE=1
  fi
  if [ "$EXP073_DONE" -eq 0 ] && grep -q "total time:" exp073_probe.log 2>/dev/null; then
    echo "=== exp073 probe finished ==="
    cat exp073_probe.log
    EXP073_DONE=1
  fi
  sleep 20
done
echo "BOTH DONE"
