#!/bin/bash
while kill -0 229478 2>/dev/null || kill -0 229774 2>/dev/null; do
  sleep 30
done
echo "=== exp077v2 (exp077_probe_v2.log) final ==="
tail -20 /root/kaggle-vibe-template/agents/claude/workspace/exp077_probe_v2.log
echo "=== exp078 (exp078_probe.log) final ==="
tail -20 /root/kaggle-vibe-template/agents/claude/workspace/exp078_probe.log
echo DONE_BOTH
