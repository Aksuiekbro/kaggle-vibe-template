#!/bin/bash
cd /root/kaggle-vibe-template/agents/claude/workspace
prev68=$(wc -l < full_cv_exp068.log)
prev72=$(wc -l < exp072_probe_run.log)
done68=0
done72=0
while true; do
  if [ "$done68" = "0" ] && ! kill -0 210004 2>/dev/null; then
    echo "exp_068 (full CV) process exited"
    tail -20 full_cv_exp068.log
    done68=1
  fi
  cur68=$(wc -l < full_cv_exp068.log)
  if [ "$cur68" != "$prev68" ]; then
    echo "exp_068 log update:"
    tail -n +$((prev68+1)) full_cv_exp068.log
    prev68=$cur68
  fi
  if [ "$done72" = "0" ] && ! kill -0 212686 2>/dev/null; then
    echo "exp_072 (probe) process exited"
    tail -20 exp072_probe_run.log
    done72=1
  fi
  cur72=$(wc -l < exp072_probe_run.log)
  if [ "$cur72" != "$prev72" ]; then
    echo "exp_072 log update:"
    tail -n +$((prev72+1)) exp072_probe_run.log
    prev72=$cur72
  fi
  if [ "$done68" = "1" ] && [ "$done72" = "1" ]; then
    echo "BOTH_DONE"
    break
  fi
  sleep 20
done
