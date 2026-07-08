#!/bin/bash
while kill -0 137871 2>/dev/null || kill -0 139992 2>/dev/null; do
  sleep 20
done
echo "both exp042 (TTA probe) and exp043 (XGBoost+aug probe) finished"
