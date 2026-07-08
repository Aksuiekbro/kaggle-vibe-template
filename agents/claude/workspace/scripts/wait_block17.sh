#!/bin/bash
while kill -0 127709 2>/dev/null || kill -0 127710 2>/dev/null || kill -0 132003 2>/dev/null; do
  sleep 15
done
echo "=== all block-17 jobs done ==="
echo "--- exp038.log ---"; tail -10 exp038.log
echo "--- exp038_probe.log ---"; tail -10 exp038_probe.log
echo "--- exp040_probe.log ---"; tail -20 exp040_probe.log
