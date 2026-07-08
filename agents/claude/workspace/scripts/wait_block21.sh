#!/bin/bash
# wait for exp_041 full CV (pid 142264) and exp_043 catboost probe (pid 145387)
while kill -0 142264 2>/dev/null || kill -0 145387 2>/dev/null; do
  sleep 20
done
echo "DONE: both exp_041 and exp_043 finished"
tail -5 full_cv_exp041.log
echo "---"
tail -20 /root/kaggle-vibe-template/exp043_catboost_probe.log 2>/dev/null || find /root/kaggle-vibe-template -name "exp043_catboost_probe.log" -exec tail -20 {} \;
