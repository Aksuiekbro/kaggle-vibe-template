#!/bin/bash
while true; do
  if ! ps -p 199627 > /dev/null 2>&1 && ! ps -p 203372 > /dev/null 2>&1; then
    echo "both jobs finished"
    break
  fi
  sleep 20
done
echo "=== exp069_probe.log ==="
cat /root/kaggle-vibe-template/agents/claude/workspace/exp069_probe.log
echo "=== exp071_probe.log ==="
cat /root/kaggle-vibe-template/agents/claude/workspace/exp071_probe.log
