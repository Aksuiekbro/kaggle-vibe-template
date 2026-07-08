#!/bin/bash
while kill -0 163921 2>/dev/null || kill -0 164471 2>/dev/null; do
  sleep 15
done
echo "both jobs finished"
tail -20 /root/kaggle-vibe-template/agents/claude/workspace/exp053_probe.log
echo "---"
tail -20 /root/kaggle-vibe-template/agents/claude/workspace/scripts/full_cv_exp051.log
