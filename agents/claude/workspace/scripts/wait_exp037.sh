#!/bin/bash
while ps -p 119398 > /dev/null 2>&1; do
  sleep 20
done
echo "EXP037_DONE"
tail -40 exp037_probe.log
