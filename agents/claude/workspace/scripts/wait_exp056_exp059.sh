#!/bin/bash
off1=$(wc -c < agents/claude/workspace/full_cv_exp056.log)
off2=$(wc -c < agents/claude/workspace/exp059_probe.log)
while kill -0 176397 2>/dev/null || kill -0 177314 2>/dev/null; do
  sleep 15
  newoff1=$(wc -c < agents/claude/workspace/full_cv_exp056.log)
  if [ "$newoff1" -gt "$off1" ]; then
    tail -c +$((off1+1)) agents/claude/workspace/full_cv_exp056.log
    off1=$newoff1
  fi
  newoff2=$(wc -c < agents/claude/workspace/exp059_probe.log)
  if [ "$newoff2" -gt "$off2" ]; then
    tail -c +$((off2+1)) agents/claude/workspace/exp059_probe.log
    off2=$newoff2
  fi
done
echo BOTH_JOBS_DONE
