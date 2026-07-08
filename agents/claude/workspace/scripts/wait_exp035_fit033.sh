#!/bin/bash
while ps -p 115415 > /dev/null 2>&1 || ps -p 118263 > /dev/null 2>&1; do
  sleep 20
done
echo "=== exp035_full ==="
tail -20 exp035_full.log
echo "=== fit_full_exp033 ==="
tail -20 fit_full_exp033_out.log
echo "ALL_DONE"
