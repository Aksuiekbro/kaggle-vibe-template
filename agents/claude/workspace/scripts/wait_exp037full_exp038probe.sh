#!/bin/bash
while pgrep -f "exp038_fullcv_more_oversample.py" > /dev/null || pgrep -f "probe_exp038_combined_augment.py" > /dev/null; do
  sleep 15
done
echo "both jobs finished"
