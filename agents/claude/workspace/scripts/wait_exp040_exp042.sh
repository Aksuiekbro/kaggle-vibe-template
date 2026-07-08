#!/bin/bash
while kill -0 135847 2>/dev/null || kill -0 137871 2>/dev/null; do
  sleep 20
done
echo "both exp_040 (xgboost full CV) and exp_042 (TTA probe) finished"
