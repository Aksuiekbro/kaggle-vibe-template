#!/bin/bash
cd /root/kaggle-vibe-template/agents/claude/workspace
tail -n0 -f exp069_probe.log exp070_probe.log &
TAILPID=$!
while kill -0 199627 2>/dev/null || kill -0 199767 2>/dev/null; do
  sleep 5
done
sleep 2
echo "BOTH_PROBES_DONE"
kill "$TAILPID" 2>/dev/null
